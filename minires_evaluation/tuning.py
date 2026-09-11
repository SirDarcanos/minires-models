"""Deterministic, leakage-resistant candidate tuning at one orchestration seam."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from .evaluation import EvaluationConfig
from .ingestion import (
    CanonicalRow, Dataset, InputError, TRANSFORMATION_VERSION, fingerprint,
    load_records, normalize,
)
from .learned import LearnedBaselineConfig, _matrix, _predict
from .legacy import LEGACY_FEATURES
from .private_io import create_private_file, write_private_json
from .splits import freeze_splits


TUNING_VERSION = "minires-candidate-tuning-v1"

NEURAL_NETWORK_DOMAIN: dict[str, tuple[Any, ...]] = {
    "layers": ((64, 32), (128, 64), (128, 64, 32), (256, 128, 64),
               (384, 192, 96), (512, 256, 128, 64)),
    "activation": ("relu", "selu", "mish"),
    "dropout": (0.0, 0.1, 0.2, 0.3),
    "optimizer": ("adam", "adamw"),
    "loss": ("mean_absolute_error", "huber", "mean_squared_error"),
    "learning_rate": (0.0001, 0.0003, 0.001, 0.003),
    "l2": (0.0, 1e-6, 1e-5, 1e-4),
    "batch_size": (32, 64, 128, 256),
    "maximum_epochs": (100,),
    "early_stopping_patience": (8,),
}

XGBOOST_DOMAIN: dict[str, tuple[Any, ...]] = {
    "n_estimators": (300, 600, 900, 1200),
    "max_depth": (3, 5, 7, 9),
    "learning_rate": (0.01, 0.03, 0.05, 0.1),
    "subsample": (0.6, 0.75, 0.9, 1.0),
    "colsample_bytree": (0.6, 0.75, 0.9, 1.0),
    "min_child_weight": (1.0, 3.0, 6.0, 10.0),
    "gamma": (0.0, 0.05, 0.2, 0.5),
    "reg_alpha": (0.0, 1e-5, 1e-3, 0.1),
    "reg_lambda": (0.1, 1.0, 5.0, 10.0),
    "objective": ("reg:squarederror",),
    "n_jobs": (1,),
    "early_stopping_rounds": (50,),
}


@dataclass(frozen=True)
class SearchLimits:
    seed: int
    second_seed: int | None = None
    maximum_candidate_runs: int = 20
    maximum_elapsed_seconds: float = 7200.0
    neural_network_trials: int = 6
    xgboost_trials: int = 6
    ensemble_trials: int = 3
    second_seed_candidates: int = 5
    ensemble_neural_network_weights: tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    )

    @property
    def resolved_second_seed(self) -> int:
        return self.seed + 1 if self.second_seed is None else self.second_seed


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class SearchPlan:
    version: str
    seed: int
    second_seed: int
    input_fingerprint: str
    code_fingerprint: str
    dependency_versions: dict[str, str]
    parameter_domains: dict[str, dict[str, tuple[Any, ...]]]
    component_trials: tuple[Candidate, ...]
    ensemble_rule: dict[str, Any]
    ranking_rule: tuple[str, ...]
    resource_limits: dict[str, int | float]
    control: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateFoldFit:
    predictor: Callable[[Sequence[tuple[float, ...]]], Sequence[float]]
    metadata: dict[str, Any]
    fitted_state: dict[str, Any]


@dataclass(frozen=True)
class LockedFit:
    predictor: Callable[[Sequence[tuple[float, ...]]], Sequence[float]]
    preprocessing_state: dict[str, Any]
    artifacts: dict[str, bytes]
    metadata: dict[str, Any]


class CandidateRuntime(Protocol):
    dependency_versions: dict[str, str]

    def fit_fold(
        self, candidate: Candidate, seed: int,
        train_features: Sequence[tuple[float, ...]], train_targets: Sequence[float],
        validation_features: Sequence[tuple[float, ...]], validation_targets: Sequence[float],
    ) -> CandidateFoldFit: ...

    def refit(
        self, candidate: Candidate, seed: int,
        features: Sequence[tuple[float, ...]], targets: Sequence[float],
        fixed_training_counts: Mapping[str, int | float],
    ) -> LockedFit: ...

    def load_locked(
        self, candidate: Candidate, directory: Path, contract: Mapping[str, Any]
    ) -> Callable[[Sequence[tuple[float, ...]]], Sequence[float]]: ...


@dataclass(frozen=True)
class CandidateRun:
    candidate: Candidate
    seed: int
    status: str
    blockers: tuple[str, ...]
    eligible: bool
    metrics: dict[str, float | int | None]
    source_reports: tuple[dict[str, Any], ...]
    fit_metadata: tuple[dict[str, Any], ...]
    resource_use: dict[str, Any]


@dataclass(frozen=True)
class LockedCandidate:
    candidate: Candidate
    predictor: Callable[[Sequence[tuple[float, ...]]], Sequence[float]]
    directory: Path
    manifest: dict[str, Any]
    contract: dict[str, Any]


@dataclass(frozen=True)
class TuningResult:
    status: str
    blockers: tuple[str, ...]
    run_count: int
    allocation: dict[str, int]
    plan: SearchPlan
    control_result: CandidateRun | None
    initial_results: tuple[CandidateRun, ...]
    second_seed_results: tuple[CandidateRun, ...]
    combined_results: tuple[dict[str, Any], ...]
    locked_candidate: LockedCandidate | None
    resource_use: dict[str, Any]

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        base: dict[str, Any] = {
            "status": self.status,
            "blockers": list(self.blockers),
            "run_count": self.run_count,
            "planned_candidate_runs": self.plan.resource_limits["maximum_candidate_runs"],
            "skipped_candidate_runs": (
                int(self.plan.resource_limits["maximum_candidate_runs"]) - self.run_count
            ),
            "allocation": dict(self.allocation),
            "resource_use": dict(self.resource_use),
            "candidate_selected": self.locked_candidate is not None,
            "classification": "internal_advisory_candidate_tuning",
            "limitations": [
                "human_review_required_for_every_estimate",
                "no_population_wide_performance_claim",
                "no_prediction_interval",
                "no_actual_shop_consumption_or_pricing_claim",
            ],
        }
        if public:
            return base
        base.update({
            "search_plan": self.plan.to_dict(),
            "control_result": _run_to_dict(self.control_result),
            "initial_results": [_run_to_dict(item) for item in self.initial_results],
            "second_seed_results": [_run_to_dict(item) for item in self.second_seed_results],
            "combined_results": list(self.combined_results),
            "skipped_candidates": _skipped_candidates(self),
            "locked_candidate": self.locked_candidate.contract if self.locked_candidate else None,
        })
        return base


def verify_locked_candidate_files(
    directory: str | Path,
    dependency_versions: Mapping[str, str],
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_contract: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, ...], dict[str, Any], dict[str, Any]]:
    """Verify the one authoritative lock file, checksum, and dependency contract."""
    root = Path(directory)
    blockers: set[str] = set()
    manifest: dict[str, Any] = {}
    contract: dict[str, Any] = {}
    try:
        manifest = json.loads((root / "lock-manifest.json").read_text())
        contract = json.loads((root / "candidate-contract.json").read_text())
        if manifest.get("version") != TUNING_VERSION:
            blockers.add("locked_candidate_manifest_mismatch")
        if expected_manifest is not None and manifest != json.loads(json.dumps(expected_manifest)):
            blockers.add("locked_candidate_manifest_mismatch")
        listed_files = manifest.get("files", {})
        actual_files = {path.name for path in root.iterdir()
                        if path.is_file() and path.name != "lock-manifest.json"}
        if (not isinstance(listed_files, dict) or set(listed_files) != actual_files
                or not {"candidate-contract.json", "preprocessing-state.json"}.issubset(actual_files)
                or len(actual_files) < 3):
            blockers.add("locked_candidate_manifest_mismatch")
        if isinstance(listed_files, dict):
            for name, checksum in listed_files.items():
                path = root / name
                if (not isinstance(name, str) or not isinstance(checksum, str)
                        or not path.is_file()
                        or sha256(path.read_bytes()).hexdigest() != checksum):
                    blockers.add("locked_candidate_checksum_mismatch")
        if expected_contract is not None and contract != json.loads(json.dumps(expected_contract)):
            blockers.add("locked_candidate_contract_mismatch")
        if contract.get("dependency_versions") != dict(sorted(dependency_versions.items())):
            blockers.add("locked_candidate_dependency_mismatch")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        blockers.add("locked_candidate_unavailable")
    return tuple(sorted(blockers)), manifest, contract


def load_locked_candidate(
    directory: str | Path, runtime: CandidateRuntime
) -> LockedCandidate:
    """Load a checksum-verified fitted candidate for a later assessment process."""
    root = Path(directory)
    blockers, manifest, contract = verify_locked_candidate_files(
        root, runtime.dependency_versions
    )
    if blockers:
        raise InputError(blockers[0])
    try:
        candidate = _candidate_from_dict(contract["candidate"])
        predictor = runtime.load_locked(candidate, root, contract)
        return LockedCandidate(candidate, predictor, root, manifest, contract)
    except (OSError, ValueError, TypeError, KeyError):
        raise InputError("locked_candidate_unavailable") from None


def generate_search_plan(
    limits: SearchLimits,
    *,
    input_fingerprint: str,
    code_fingerprint: str,
    dependency_versions: Mapping[str, str],
) -> SearchPlan:
    """Generate the complete finite component plan without inspecting labels."""
    _validate_limits(limits)
    neural = _space_filling_candidates(
        "neural_network", NEURAL_NETWORK_DOMAIN, limits.neural_network_trials, limits.seed
    )
    xgboost = _space_filling_candidates(
        "xgboost", XGBOOST_DOMAIN, limits.xgboost_trials, limits.seed ^ 0x5EED
    )
    return SearchPlan(
        version=TUNING_VERSION,
        seed=limits.seed,
        second_seed=limits.resolved_second_seed,
        input_fingerprint=input_fingerprint,
        code_fingerprint=code_fingerprint,
        dependency_versions=dict(sorted(dependency_versions.items())),
        parameter_domains={
            "neural_network": NEURAL_NETWORK_DOMAIN,
            "xgboost": XGBOOST_DOMAIN,
        },
        component_trials=neural + xgboost,
        ensemble_rule={
            "trial_count": limits.ensemble_trials,
            "pairing": "rank_each_component_family_then_pair_equal_rank",
            "weight_selection_partition": "development_validation_only",
            "neural_network_weight_grid": limits.ensemble_neural_network_weights,
        },
        ranking_rule=(
            "eligible_serious_error_gates_first",
            "source_balanced_mae_g_ascending",
            "pooled_mae_g_ascending",
            "pooled_within_2g_fraction_descending",
            "stable_candidate_id_ascending",
        ),
        resource_limits={
            "maximum_candidate_runs": limits.maximum_candidate_runs,
            "maximum_elapsed_seconds": limits.maximum_elapsed_seconds,
            "initial_neural_network_trials": limits.neural_network_trials,
            "initial_xgboost_trials": limits.xgboost_trials,
            "initial_ensemble_trials": limits.ensemble_trials,
            "second_seed_candidates": limits.second_seed_candidates,
        },
        control={
            "classification": "clean_fixed_configuration_baseline",
            "candidate_slot_consumed": False,
            "configuration_mutable": False,
        },
    )


def _space_filling_candidates(
    family: str, domain: Mapping[str, tuple[Any, ...]], count: int, seed: int
) -> tuple[Candidate, ...]:
    rng = random.Random(seed)
    keys = tuple(domain)
    offsets = {key: rng.randrange(len(domain[key])) for key in keys}
    steps = {key: rng.choice([step for step in range(1, len(domain[key]) + 1)
                             if math.gcd(step, len(domain[key])) == 1]) for key in keys}
    candidates = []
    for trial in range(count):
        parameters = {
            key: domain[key][(offsets[key] + trial * steps[key]) % len(domain[key])]
            for key in keys
        }
        identity = json.dumps([family, parameters], sort_keys=True, separators=(",", ":"))
        candidates.append(Candidate(
            candidate_id=f"{family[:3]}-{trial + 1:02d}-{sha256(identity.encode()).hexdigest()[:8]}",
            family=family,
            parameters=parameters,
        ))
    return tuple(candidates)


def tune_candidates(
    records: Dataset,
    config: EvaluationConfig,
    *,
    runtime: CandidateRuntime,
    output_root: str | Path,
    limits: SearchLimits,
    clock: Callable[[], float] = time.monotonic,
    plan: SearchPlan | None = None,
) -> TuningResult:
    """Run the complete bounded search and lock one development-only candidate."""
    output = Path(output_root)
    if "private" not in output.resolve().parts:
        raise InputError("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None

    loaded, input_fingerprint = load_records(records)
    rows = normalize(loaded, config, contract="legacy")
    code_fingerprint = fingerprint({
        path.name: path.read_text() for path in sorted(Path(__file__).parent.glob("*.py"))
    })
    selected_plan = plan or generate_search_plan(
        limits,
        input_fingerprint=input_fingerprint,
        code_fingerprint=code_fingerprint,
        dependency_versions=runtime.dependency_versions,
    )
    try:
        _validate_plan(selected_plan, limits, input_fingerprint, code_fingerprint,
                       runtime.dependency_versions)
    except ValueError:
        evidence_plan = generate_search_plan(
            limits, input_fingerprint=input_fingerprint,
            code_fingerprint=code_fingerprint,
            dependency_versions=runtime.dependency_versions,
        )
        result = TuningResult(
            "blocked", ("invalid_search_plan",), 0,
            {"neural_network": 0, "xgboost": 0, "ensemble": 0,
             "second_seed": 0, "control": 0},
            evidence_plan, None, (), (), (), None, _resources(0.0, 0.0),
        )
        _write_tuning_outputs(output, result)
        return result
    # Create the complete plan before either split allocation or runtime fitting.
    write_private_json(output / "search-plan.json", selected_plan.to_dict())
    startup_blockers = tuple(getattr(runtime, "startup_blockers", ()))
    if startup_blockers:
        result = TuningResult(
            "blocked", tuple(sorted(set(startup_blockers))), 0,
            {"neural_network": 0, "xgboost": 0, "ensemble": 0,
             "second_seed": 0, "control": 0},
            selected_plan, None, (), (), (), None, _resources(0.0, 0.0),
        )
        _write_tuning_outputs(output, result)
        return result
    manifest = freeze_splits(
        rows, input_fingerprint, asdict(config), output / "development-splits.json"
    )
    if manifest["status"] != "frozen_source_holdout":
        result = TuningResult(
            "blocked", tuple(manifest["blockers"]), 0,
            {"neural_network": 0, "xgboost": 0, "ensemble": 0,
             "second_seed": 0, "control": 0},
            selected_plan, None, (), (), (), None, _resources(0.0, 0.0),
        )
        _write_tuning_outputs(output, result)
        return result

    by_index = {row.row_index: row for row in rows if row.outcome == "included"}
    started = clock()
    cpu_started = time.process_time()
    deadline = started + limits.maximum_elapsed_seconds
    allocation = {"neural_network": 0, "xgboost": 0, "ensemble": 0,
                  "second_seed": 0, "control": 0}
    run_count = 0
    blockers: list[str] = []

    control = Candidate("clean-fixed-control", "control", asdict(LearnedBaselineConfig()))
    control_result = _evaluate_candidate(
        control, LearnedBaselineConfig().seed, runtime, manifest, by_index,
        limits.ensemble_neural_network_weights,
    )
    allocation["control"] = 1
    if control_result.status != "completed":
        blockers.append("control_evaluation_failed")

    initial: list[CandidateRun] = []
    for candidate in selected_plan.component_trials if not blockers else ():
        now = clock()
        if run_count >= limits.maximum_candidate_runs or now >= deadline:
            blockers.append("candidate_search_deadline_reached" if now >= deadline
                            else "candidate_run_limit_reached")
            break
        candidate_result = _evaluate_candidate(
            candidate, selected_plan.seed, runtime, manifest, by_index,
            limits.ensemble_neural_network_weights,
        )
        initial.append(candidate_result)
        allocation[candidate.family] += 1
        run_count += 1
        if candidate_result.status != "completed":
            blockers.append("candidate_runtime_failed")
            break

    if len(initial) == len(selected_plan.component_trials) and not blockers:
        neural = [item for item in initial if item.candidate.family == "neural_network"]
        xgboost = [item for item in initial if item.candidate.family == "xgboost"]
        for index in range(limits.ensemble_trials):
            now = clock()
            if run_count >= limits.maximum_candidate_runs or now >= deadline:
                blockers.append("candidate_search_deadline_reached" if now >= deadline
                                else "candidate_run_limit_reached")
                break
            candidate = _ensemble_candidate(
                index, neural, xgboost, limits.ensemble_neural_network_weights
            )
            ensemble_result = _evaluate_candidate(
                candidate, selected_plan.seed, runtime, manifest, by_index,
                limits.ensemble_neural_network_weights,
            )
            initial.append(ensemble_result)
            allocation["ensemble"] += 1
            run_count += 1
            if ensemble_result.status != "completed":
                blockers.append("candidate_runtime_failed")
                break

    second: list[CandidateRun] = []
    if len(initial) == 15 and not blockers:
        ranked_initial = sorted((item for item in initial if item.eligible), key=_rank_key)
        if len(ranked_initial) < limits.second_seed_candidates:
            blockers.append("insufficient_eligible_candidates")
        else:
            for first in ranked_initial[:limits.second_seed_candidates]:
                now = clock()
                if run_count >= limits.maximum_candidate_runs or now >= deadline:
                    blockers.append("candidate_search_deadline_reached" if now >= deadline
                                    else "candidate_run_limit_reached")
                    break
                repeated_result = _evaluate_candidate(
                    first.candidate, selected_plan.second_seed, runtime, manifest, by_index,
                    limits.ensemble_neural_network_weights,
                )
                second.append(repeated_result)
                allocation["second_seed"] += 1
                run_count += 1
                if repeated_result.status != "completed":
                    blockers.append("candidate_runtime_failed")
                    break

    combined = _combine_seed_results(initial, second)
    locked: LockedCandidate | None = None
    complete = run_count == limits.maximum_candidate_runs and len(second) == 5 and not blockers
    if complete:
        eligible = [item for item in combined if item["eligible"]]
        if not eligible:
            blockers.append("no_eligible_candidate")
        else:
            selected = min(eligible, key=_combined_rank_key)
            candidate = next(item.candidate for item in initial
                             if item.candidate.candidate_id == selected["candidate_id"])
            candidate = _resolve_final_candidate(candidate)
            try:
                locked = _refit_and_lock(
                    candidate, selected_plan, runtime, tuple(by_index.values()), initial, second,
                    manifest, output / "locked-candidate",
                )
            except Exception:
                blockers.append("candidate_refit_or_lock_failed")
    elapsed = max(0.0, clock() - started)
    resources = _resources(elapsed, time.process_time() - cpu_started)
    status = ("completed" if locked is not None else "blocked"
              if blockers or not complete else "completed_no_candidate")
    result = TuningResult(
        status, tuple(sorted(set(blockers))), run_count, allocation, selected_plan,
        control_result, tuple(initial), tuple(second), tuple(combined), locked, resources,
    )
    _write_tuning_outputs(output, result)
    return result


def _evaluate_candidate(
    candidate: Candidate, seed: int, runtime: CandidateRuntime,
    manifest: Mapping[str, Any], by_index: Mapping[int, CanonicalRow],
    weight_grid: Sequence[float],
) -> CandidateRun:
    reports: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        for fold in manifest["folds"]:
            train_x, train_y = _matrix([by_index[index] for index in fold["train"]])
            validation_x, validation_y = _matrix([by_index[index] for index in fold["validation"]])
            test_x, test_y = _matrix([by_index[index] for index in fold["test"]])
            if candidate.family == "ensemble":
                fold_components = next(
                    item for item in candidate.parameters["components_by_fold"]
                    if item["source"] == fold["source"]
                )
                neural = _candidate_from_dict(fold_components["neural_network"])
                xgboost = _candidate_from_dict(fold_components["xgboost"])
                neural_fit = runtime.fit_fold(neural, seed, train_x, train_y, validation_x, validation_y)
                xgboost_fit = runtime.fit_fold(xgboost, seed, train_x, train_y, validation_x, validation_y)
                validation_neural = _predict(neural_fit.predictor, validation_x)
                validation_xgboost = _predict(xgboost_fit.predictor, validation_x)
                weight = _select_ensemble_weight(validation_y, validation_neural,
                                                 validation_xgboost, weight_grid)
                test_neural = _predict(neural_fit.predictor, test_x)
                test_xgboost = _predict(xgboost_fit.predictor, test_x)
                predictions = tuple(weight * left + (1 - weight) * right
                                    for left, right in zip(test_neural, test_xgboost))
                fold_metadata = {
                    "neural_network": neural_fit.metadata,
                    "xgboost": xgboost_fit.metadata,
                    "selected_neural_network_weight": weight,
                    "fitted_state_fingerprint": fingerprint({
                        "neural_network": _fingerprintable_state(neural_fit.fitted_state),
                        "xgboost": _fingerprintable_state(xgboost_fit.fitted_state),
                        "weight": weight,
                    }),
                }
            else:
                fitted = runtime.fit_fold(candidate, seed, train_x, train_y,
                                          validation_x, validation_y)
                validation_predictions = _predict(fitted.predictor, validation_x)
                predictions = _predict(fitted.predictor, test_x)
                fold_metadata = {
                    **fitted.metadata,
                    "fitted_state_fingerprint": fingerprint(
                        _fingerprintable_state(fitted.fitted_state)
                    ),
                }
            reports.append({
                "source": fold["source"],
                "test_rows": list(fold["test"]),
                "train_rows": list(fold["train"]),
                "validation_rows": list(fold["validation"]),
                "actual": list(test_y),
                "predictions": list(predictions),
                **({
                    "validation_actual": list(validation_y),
                    "validation_predictions": list(validation_predictions),
                } if candidate.family != "ensemble" else {}),
                "test_data_fingerprint": fingerprint({"features": test_x, "targets": test_y}),
            })
            metadata.append(fold_metadata)
        metrics = _candidate_metrics(reports)
        eligible = _tail_eligible(metrics)
        blockers = () if eligible else ("development_serious_error_gate_failed",)
        return CandidateRun(
            candidate, seed, "completed", blockers, eligible, metrics,
            tuple(reports), tuple(metadata),
            _resources(time.perf_counter() - started, time.process_time() - cpu_started),
        )
    except Exception:
        return CandidateRun(
            candidate, seed, "blocked", ("candidate_runtime_failed",), False,
            _empty_metrics(), (), (),
            _resources(time.perf_counter() - started, time.process_time() - cpu_started),
        )


def _fingerprintable_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if not key.startswith("_")}


def _candidate_metrics(reports: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    source_metrics = []
    all_errors: list[float] = []
    for report in reports:
        errors = [abs(float(prediction) - float(actual))
                  for prediction, actual in zip(report["predictions"], report["actual"])]
        all_errors.extend(errors)
        source_metrics.append({
            "mae": math.fsum(errors) / len(errors),
            "within": sum(error <= 2.0 for error in errors) / len(errors),
            "tail": sum(error > 5.0 for error in errors) / len(errors),
        })
    count = len(all_errors)
    source_count = len(source_metrics)
    return {
        "sample_count": count,
        "source_count": source_count,
        "pooled_mae_g": math.fsum(all_errors) / count if count else None,
        "source_balanced_mae_g": (math.fsum(item["mae"] for item in source_metrics) / source_count
                                  if source_count else None),
        "pooled_within_2g_fraction": (sum(error <= 2.0 for error in all_errors) / count
                                      if count else None),
        "source_balanced_within_2g_fraction": (
            math.fsum(item["within"] for item in source_metrics) / source_count
            if source_count else None
        ),
        "pooled_above_5g_fraction": (sum(error > 5.0 for error in all_errors) / count
                                     if count else None),
        "source_balanced_above_5g_fraction": (
            math.fsum(item["tail"] for item in source_metrics) / source_count
            if source_count else None
        ),
        "maximum_source_above_5g_fraction": (
            max(item["tail"] for item in source_metrics) if source_metrics else None
        ),
    }


def _empty_metrics() -> dict[str, float | int | None]:
    return {key: 0 if key in {"sample_count", "source_count"} else None for key in (
        "sample_count", "source_count", "pooled_mae_g", "source_balanced_mae_g",
        "pooled_within_2g_fraction", "source_balanced_within_2g_fraction",
        "pooled_above_5g_fraction", "source_balanced_above_5g_fraction",
        "maximum_source_above_5g_fraction",
    )}


def _tail_eligible(metrics: Mapping[str, Any]) -> bool:
    return (
        metrics["pooled_above_5g_fraction"] is not None
        and metrics["pooled_above_5g_fraction"] <= 0.01
        and metrics["source_balanced_above_5g_fraction"] <= 0.01
        and metrics["maximum_source_above_5g_fraction"] <= 0.02
    )


def _rank_key(run: CandidateRun) -> tuple[Any, ...]:
    metrics = run.metrics
    return (
        0 if run.eligible else 1,
        metrics["source_balanced_mae_g"] if metrics["source_balanced_mae_g"] is not None else math.inf,
        metrics["pooled_mae_g"] if metrics["pooled_mae_g"] is not None else math.inf,
        -(metrics["pooled_within_2g_fraction"] or 0.0),
        run.candidate.candidate_id,
    )


def _ensemble_candidate(
    index: int, neural: Sequence[CandidateRun], xgboost: Sequence[CandidateRun],
    weights: Sequence[float],
) -> Candidate:
    sources = sorted({report["source"] for run in neural for report in run.source_reports})
    components_by_fold = []
    for source in sources:
        ranked_neural = sorted(neural, key=lambda run: _validation_rank_key(run, source))
        ranked_xgboost = sorted(xgboost, key=lambda run: _validation_rank_key(run, source))
        components_by_fold.append({
            "source": source,
            "neural_network": asdict(ranked_neural[index].candidate),
            "xgboost": asdict(ranked_xgboost[index].candidate),
        })
    parameters = {
        "component_rank": index + 1,
        "selection_partition": "fold_validation_only",
        "components_by_fold": tuple(components_by_fold),
        "neural_network_weight_grid": tuple(weights),
    }
    digest = sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:8]
    return Candidate(f"ens-{index + 1:02d}-{digest}", "ensemble", parameters)


def _validation_rank_key(run: CandidateRun, source: str) -> tuple[float, str]:
    report = next(item for item in run.source_reports if item["source"] == source)
    actual = report["validation_actual"]
    predictions = report["validation_predictions"]
    mae = math.fsum(abs(float(prediction) - float(target))
                     for prediction, target in zip(predictions, actual)) / len(actual)
    return mae, run.candidate.candidate_id


def _resolve_final_candidate(candidate: Candidate) -> Candidate:
    if candidate.family != "ensemble" or "components_by_fold" not in candidate.parameters:
        return candidate
    pairings: dict[str, tuple[int, dict[str, Any]]] = {}
    for item in candidate.parameters["components_by_fold"]:
        key = (item["neural_network"]["candidate_id"] + ":"
               + item["xgboost"]["candidate_id"])
        count, _ = pairings.get(key, (0, item))
        pairings[key] = (count + 1, item)
    _, selected = min(pairings.values(), key=lambda value: (-value[0],
        value[1]["neural_network"]["candidate_id"],
        value[1]["xgboost"]["candidate_id"]))
    return Candidate(candidate.candidate_id, candidate.family, {
        "neural_network": selected["neural_network"],
        "xgboost": selected["xgboost"],
        "neural_network_weight_grid": candidate.parameters["neural_network_weight_grid"],
        "component_resolution": "most_frequent_fold_validation_pair_then_stable_id",
    })


def _candidate_from_dict(value: Mapping[str, Any]) -> Candidate:
    parameters = _freeze_json_lists(value["parameters"])
    if not isinstance(parameters, dict):
        raise ValueError("invalid candidate parameters")
    return Candidate(str(value["candidate_id"]), str(value["family"]), parameters)


def _freeze_json_lists(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_json_lists(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json_lists(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _freeze_json_lists(item) for key, item in value.items()}
    return value


def _select_ensemble_weight(actual: Sequence[float], neural: Sequence[float],
                            xgboost: Sequence[float], weights: Sequence[float]) -> float:
    return min(weights, key=lambda weight: (
        math.fsum(abs(weight * left + (1 - weight) * right - target)
                  for left, right, target in zip(neural, xgboost, actual)) / len(actual),
        weight,
    ))


def _combine_seed_results(initial: Sequence[CandidateRun], second: Sequence[CandidateRun]
                          ) -> list[dict[str, Any]]:
    by_id = {item.candidate.candidate_id: item for item in initial}
    combined = []
    for repeated in second:
        first = by_id[repeated.candidate.candidate_id]
        metric_keys = first.metrics.keys()
        metrics: dict[str, float | None] = {}
        for key in metric_keys:
            left = first.metrics[key]
            right = repeated.metrics[key]
            metrics[key] = ((float(left) + float(right)) / 2
                            if isinstance(left, (int, float))
                            and isinstance(right, (int, float)) else None)
        combined.append({
            "candidate_id": first.candidate.candidate_id,
            "family": first.candidate.family,
            "eligible": first.eligible and repeated.eligible,
            "seed_results": [first.seed, repeated.seed],
            "equal_seed_weight": 0.5,
            "metrics": metrics,
        })
    return combined


def _combined_rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = item["metrics"]
    return (metrics["source_balanced_mae_g"], metrics["pooled_mae_g"],
            -metrics["pooled_within_2g_fraction"], item["candidate_id"])


def _refit_and_lock(
    candidate: Candidate, plan: SearchPlan, runtime: CandidateRuntime,
    rows: Sequence[CanonicalRow], initial: Sequence[CandidateRun],
    second: Sequence[CandidateRun], development_manifest: Mapping[str, Any], directory: Path,
) -> LockedCandidate:
    related = [run for run in (*initial, *second)
               if run.candidate.candidate_id == candidate.candidate_id]
    fixed_counts = _derive_training_counts(candidate, related)
    features, targets = _matrix(rows)
    fitted = runtime.refit(candidate, plan.second_seed, features, targets, fixed_counts)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    contract = {
        "version": TUNING_VERSION,
        "candidate": asdict(candidate),
        "input_fingerprint": plan.input_fingerprint,
        "code_fingerprint": plan.code_fingerprint,
        "search_plan_fingerprint": fingerprint(plan.to_dict()),
        "development_split_fingerprint": fingerprint(development_manifest),
        "transformation_version": TRANSFORMATION_VERSION,
        "features": list(LEGACY_FEATURES),
        "preprocessing": "locked_fit_on_all_eligible_development_records_only",
        "eligibility_rule": {
            "pooled_above_5g_fraction_maximum": 0.01,
            "source_balanced_above_5g_fraction_maximum": 0.01,
            "per_source_above_5g_fraction_maximum": 0.02,
        },
        "ranking_rule": list(plan.ranking_rule),
        "dependency_versions": dict(plan.dependency_versions),
        "fixed_training_counts": fixed_counts,
        "refit_partition": "all_eligible_development_records",
        "final_test_access": False,
        "classification": "internal_advisory_human_review_required",
        "output_unit": "g",
        "runtime_metadata": fitted.metadata,
    }
    write_private_json(directory / "candidate-contract.json", contract)
    write_private_json(directory / "preprocessing-state.json", fitted.preprocessing_state)
    for name, content in sorted(fitted.artifacts.items()):
        if not name or Path(name).name != name or not isinstance(content, bytes):
            raise InputError("invalid_locked_candidate_artifact")
        with create_private_file(directory / name) as stream:
            stream.write(content)
    files = sorted(path for path in directory.iterdir() if path.is_file())
    manifest = {
        "version": TUNING_VERSION,
        "files": {path.name: sha256(path.read_bytes()).hexdigest() for path in files},
        "locked_before_final_assessment": True,
    }
    write_private_json(directory / "lock-manifest.json", manifest)
    return LockedCandidate(candidate, fitted.predictor, directory, manifest, contract)


def _derive_training_counts(candidate: Candidate, runs: Sequence[CandidateRun]) -> dict[str, int | float]:
    epochs: list[int] = []
    trees: list[int] = []
    weights: list[float] = []
    for run in runs:
        for metadata in run.fit_metadata:
            values = metadata.values() if candidate.family == "ensemble" else (metadata,)
            for value in values:
                if isinstance(value, Mapping):
                    if isinstance(value.get("selected_epochs"), int):
                        epochs.append(value["selected_epochs"])
                    if isinstance(value.get("selected_trees"), int):
                        trees.append(value["selected_trees"])
            weight = metadata.get("selected_neural_network_weight")
            if isinstance(weight, (int, float)) and math.isfinite(weight):
                weights.append(float(weight))
    result: dict[str, int | float] = {}
    if epochs:
        result["neural_network_epochs"] = int(statistics.median(epochs))
    if trees:
        result["xgboost_trees"] = int(statistics.median(trees))
    if weights:
        result["ensemble_neural_network_weight"] = statistics.median(weights)
    return result


def _validate_plan(plan: SearchPlan, limits: SearchLimits, input_fingerprint: str,
                   code_fingerprint: str, dependencies: Mapping[str, str]) -> None:
    _validate_limits(limits)
    expected_plan = generate_search_plan(
        limits, input_fingerprint=input_fingerprint, code_fingerprint=code_fingerprint,
        dependency_versions=dependencies,
    )
    if plan != expected_plan:
        raise ValueError("invalid_search_plan")
    if (plan.version != TUNING_VERSION or plan.seed != limits.seed
            or plan.second_seed != limits.resolved_second_seed
            or plan.input_fingerprint != input_fingerprint
            or plan.code_fingerprint != code_fingerprint
            or plan.dependency_versions != dict(sorted(dependencies.items()))):
        raise ValueError("invalid_search_plan")
    families = [candidate.family for candidate in plan.component_trials]
    if families.count("neural_network") != 6 or families.count("xgboost") != 6:
        raise ValueError("invalid_search_plan")
    if len({candidate.candidate_id for candidate in plan.component_trials}) != 12:
        raise ValueError("invalid_search_plan")
    for candidate in plan.component_trials:
        domain = plan.parameter_domains.get(candidate.family)
        if domain is None or set(candidate.parameters) != set(domain):
            raise ValueError("invalid_search_plan")
        for key, value in candidate.parameters.items():
            if value not in domain[key]:
                raise ValueError("invalid_search_plan")
        if candidate.family == "xgboost" and (
            candidate.parameters["objective"] != "reg:squarederror"
            or candidate.parameters["n_jobs"] != 1
        ):
            raise ValueError("invalid_search_plan")


def _write_tuning_outputs(output: Path, result: TuningResult) -> None:
    write_private_json(output / "tuning-result.json", result.to_dict())
    public = result.to_dict(public=True)
    public["publication_status"] = "draft_not_approved"
    write_private_json(output / "public-summary-draft.json", public)
    from .evidence import review_public_summary
    write_private_json(output / "public-summary-review.json", review_public_summary(public))
    inventory = {
        str(path.relative_to(output)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_private_json(output / "manifest.json", {
        "version": TUNING_VERSION,
        "artifacts": inventory,
        "publication_performed": False,
    })


def _skipped_candidates(result: TuningResult) -> list[dict[str, Any]]:
    attempted = {run.candidate.candidate_id for run in result.initial_results}
    skipped = [
        {"candidate_id": candidate.candidate_id, "phase": "initial", "reason": "search_stopped"}
        for candidate in result.plan.component_trials if candidate.candidate_id not in attempted
    ]
    skipped.extend(
        {"candidate_id": f"ensemble-slot-{index + 1:02d}", "phase": "initial",
         "reason": "search_stopped"}
        for index in range(result.allocation["ensemble"], 3)
    )
    skipped.extend(
        {"candidate_id": f"second-seed-slot-{index + 1:02d}", "phase": "second_seed",
         "reason": "search_stopped"}
        for index in range(result.allocation["second_seed"], 5)
    )
    return skipped[:max(0, 20 - result.run_count)]


def _run_to_dict(run: CandidateRun | None) -> dict[str, Any] | None:
    if run is None:
        return None
    return {
        "candidate": asdict(run.candidate), "seed": run.seed, "status": run.status,
        "blockers": list(run.blockers), "eligible": run.eligible,
        "metrics": dict(run.metrics), "source_reports": list(run.source_reports),
        "fit_metadata": list(run.fit_metadata), "resource_use": dict(run.resource_use),
    }


def _resources(elapsed: float, cpu: float) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "elapsed_seconds": elapsed,
        "process_cpu_seconds": max(0.0, cpu),
        "process_peak_rss_platform_units": usage.ru_maxrss,
        "peak_rss_scope": "tuning_process_high_water_mark",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


class TensorflowXGBoostCandidateRuntime:
    """Optional bounded adapter for the pinned candidate model families."""

    def __init__(self) -> None:
        if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
            raise RuntimeError("unsupported_candidate_runtime")
        import keras
        import numpy as np
        import tensorflow as tf
        import xgboost
        from .learned import enable_synchronous_dataset_execution
        if not enable_synchronous_dataset_execution():
            raise RuntimeError("synchronous_tensorflow_dataset_runtime_required")
        self.np = np
        self.tf = tf
        self.xgboost = xgboost
        self.dependency_versions = {
            "keras": keras.__version__, "numpy": np.__version__,
            "tensorflow": tf.__version__, "xgboost": xgboost.__version__,
        }

    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets):
        if candidate.family == "control":
            neural = self._fit_neural(
                _fixed_neural_parameters(candidate.parameters), seed, train_features,
                train_targets, validation_features, validation_targets,
            )
            xgboost = self._fit_xgboost(
                _fixed_xgboost_parameters(candidate.parameters), seed, train_features,
                train_targets, validation_features, validation_targets,
            )
            return CandidateFoldFit(
                predictor=lambda rows: [0.2 * left + 0.8 * right for left, right in zip(
                    neural.predictor(rows), xgboost.predictor(rows))],
                metadata={"neural_network": neural.metadata, "xgboost": xgboost.metadata,
                          "selected_neural_network_weight": 0.2},
                fitted_state={"neural_network": neural.fitted_state,
                              "xgboost": xgboost.fitted_state, "weight": 0.2},
            )
        if candidate.family == "neural_network":
            return self._fit_neural(candidate.parameters, seed, train_features, train_targets,
                                    validation_features, validation_targets)
        if candidate.family == "xgboost":
            return self._fit_xgboost(candidate.parameters, seed, train_features, train_targets,
                                     validation_features, validation_targets)
        raise ValueError("unsupported_candidate_family")

    def refit(self, candidate, seed, features, targets, fixed_training_counts):
        if candidate.family == "ensemble":
            neural_candidate = _candidate_from_dict(candidate.parameters["neural_network"])
            xgboost_candidate = _candidate_from_dict(candidate.parameters["xgboost"])
            neural_parameters = dict(neural_candidate.parameters)
            neural_parameters["maximum_epochs"] = fixed_training_counts["neural_network_epochs"]
            xgboost_parameters = dict(xgboost_candidate.parameters)
            xgboost_parameters["n_estimators"] = fixed_training_counts["xgboost_trees"]
            neural = self._fit_neural(neural_parameters, seed, features, targets, (), ())
            xgboost = self._fit_xgboost(xgboost_parameters, seed, features, targets, (), ())
            weight = float(fixed_training_counts["ensemble_neural_network_weight"])
            return LockedFit(
                predictor=lambda rows: [weight * left + (1 - weight) * right for left, right in zip(
                    neural.predictor(rows), xgboost.predictor(rows))],
                preprocessing_state={
                    "neural_network": neural.fitted_state.get("normalization"),
                    "xgboost": "unnormalized_float32",
                },
                artifacts={**{f"neural-{key}": value for key, value in _artifacts(neural).items()},
                           **{f"xgboost-{key}": value for key, value in _artifacts(xgboost).items()}},
                metadata={"seed": seed, "neural_network_weight": weight},
            )
        parameters = dict(candidate.parameters)
        if candidate.family == "neural_network":
            parameters["maximum_epochs"] = fixed_training_counts["neural_network_epochs"]
            fitted = self._fit_neural(parameters, seed, features, targets, (), ())
            preprocessing = fitted.fitted_state.get("normalization", {})
        elif candidate.family == "xgboost":
            parameters["n_estimators"] = fixed_training_counts["xgboost_trees"]
            fitted = self._fit_xgboost(parameters, seed, features, targets, (), ())
            preprocessing = {"xgboost": "unnormalized_float32"}
        else:
            raise ValueError("unsupported_candidate_family")
        return LockedFit(fitted.predictor, preprocessing, _artifacts(fitted), {"seed": seed})

    def load_locked(self, candidate, directory, contract):
        np, tf = self.np, self.tf

        def load_neural(path):
            model = tf.keras.models.load_model(path, compile=False)
            return lambda rows: np.asarray(
                model(np.asarray(rows, dtype=np.float32), training=False)
            ).reshape(-1).tolist()

        def load_xgboost(path):
            from xgboost import XGBRegressor
            model = XGBRegressor()
            model.load_model(path)
            return lambda rows: model.predict(np.asarray(rows, dtype=np.float32)).reshape(-1).tolist()

        if candidate.family == "neural_network":
            return load_neural(directory / "model.keras")
        if candidate.family == "xgboost":
            return load_xgboost(directory / "model.json")
        if candidate.family == "ensemble":
            neural = load_neural(directory / "neural-model.keras")
            xgboost = load_xgboost(directory / "xgboost-model.json")
            weight = float(contract["fixed_training_counts"]["ensemble_neural_network_weight"])
            return lambda rows: [weight * left + (1 - weight) * right
                                 for left, right in zip(neural(rows), xgboost(rows))]
        raise ValueError("unsupported_candidate_family")

    def _fit_neural(self, parameters, seed, train_features, train_targets,
                    validation_features, validation_targets):
        np, tf = self.np, self.tf
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        train_x = np.asarray(train_features, dtype=np.float32)
        train_y = np.asarray(train_targets, dtype=np.float32)
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(train_x)
        model = tf.keras.Sequential([tf.keras.Input(shape=(train_x.shape[1],)), normalizer])
        regularizer = tf.keras.regularizers.l2(float(parameters["l2"]))
        layer_specs = parameters.get("layer_specs") or tuple(
            (units, parameters["activation"], parameters["dropout"])
            for units in parameters["layers"]
        )
        for units, activation, dropout in layer_specs:
            model.add(tf.keras.layers.Dense(int(units), kernel_regularizer=regularizer))
            model.add(tf.keras.layers.Activation(activation))
            if dropout:
                model.add(tf.keras.layers.Dropout(float(dropout)))
        model.add(tf.keras.layers.Dense(1))
        optimizer_class = (tf.keras.optimizers.AdamW if parameters["optimizer"] == "adamw"
                           else tf.keras.optimizers.Adam)
        model.compile(optimizer=optimizer_class(float(parameters["learning_rate"])),
                      loss=parameters["loss"], metrics=[tf.keras.metrics.MeanAbsoluteError()])
        kwargs: dict[str, Any] = {"verbose": 0, "shuffle": True,
                                 "epochs": int(parameters["maximum_epochs"]),
                                 "batch_size": int(parameters["batch_size"])}
        if validation_features:
            validation_x = np.asarray(validation_features, dtype=np.float32)
            validation_y = np.asarray(validation_targets, dtype=np.float32)
            kwargs["validation_data"] = (validation_x, validation_y)
            callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor="val_mean_absolute_error", mode="min",
                min_delta=float(parameters.get("early_stopping_min_delta", 0.0)),
                patience=int(parameters["early_stopping_patience"]), restore_best_weights=True)]
            if "lr_reduction_factor" in parameters:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_mean_absolute_error", mode="min",
                    factor=float(parameters["lr_reduction_factor"]),
                    patience=int(parameters["lr_reduction_patience"]),
                    min_lr=float(parameters["minimum_learning_rate"]), verbose=0,
                ))
            kwargs["callbacks"] = callbacks
        history = model.fit(train_x, train_y, **kwargs)
        parameter_hash = sha256()
        for weights in model.get_weights():
            parameter_hash.update(np.asarray(weights).tobytes())
        state = {
            "normalization": {
                "mean": normalizer.mean.numpy().reshape(-1).tolist(),
                "variance": normalizer.variance.numpy().reshape(-1).tolist(),
            },
            "fitted_parameter_fingerprint": parameter_hash.hexdigest(),
            "_model": model,
        }
        return CandidateFoldFit(
            predictor=lambda rows: np.asarray(model(np.asarray(rows, dtype=np.float32), training=False)
                                                   ).reshape(-1).tolist(),
            metadata={"selected_epochs": len(history.epoch)}, fitted_state=state,
        )

    def _fit_xgboost(self, parameters, seed, train_features, train_targets,
                     validation_features, validation_targets):
        np = self.np
        from xgboost import XGBRegressor
        kwargs = dict(parameters)
        early_stopping = kwargs.pop("early_stopping_rounds")
        model = XGBRegressor(**kwargs, random_state=seed, tree_method="hist", eval_metric="mae",
                             **({"early_stopping_rounds": early_stopping}
                                if validation_features else {}))
        fit_kwargs = ({"eval_set": [(np.asarray(validation_features, dtype=np.float32),
                                      np.asarray(validation_targets, dtype=np.float32))],
                       "verbose": False} if validation_features else {})
        model.fit(np.asarray(train_features, dtype=np.float32),
                  np.asarray(train_targets, dtype=np.float32), **fit_kwargs)
        selected = int(model.best_iteration) + 1 if validation_features else int(parameters["n_estimators"])
        return CandidateFoldFit(
            predictor=lambda rows: model.predict(np.asarray(rows, dtype=np.float32)).reshape(-1).tolist(),
            metadata={"selected_trees": selected},
            fitted_state={
                "fitted_parameter_fingerprint": sha256(
                    bytes(model.get_booster().save_raw())
                ).hexdigest(),
                "_model": model,
            },
        )


def _artifacts(fitted: CandidateFoldFit) -> dict[str, bytes]:
    model = fitted.fitted_state.get("_model")
    if model is None:
        raise ValueError("model artifact unavailable")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        if hasattr(model, "save_model"):
            path = root / "model.json"
            model.save_model(path)
        else:
            path = root / "model.keras"
            model.save(path, include_optimizer=False)
        return {path.name: path.read_bytes()}


def _fixed_neural_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "layers": (448, 601, 544, 416),
        "layer_specs": ((448, "selu", 0.0), (601, "mish", 0.3),
                        (544, "mish", 0.3), (416, "selu", 0.3)),
        "activation": "mish", "dropout": 0.3,
        "optimizer": "adamw", "loss": "mean_squared_error",
        "learning_rate": config["neural_network_learning_rate"],
        "l2": config["neural_network_l2"], "batch_size": config["batch_size"],
        "maximum_epochs": config["neural_network_max_epochs"],
        "early_stopping_patience": config["neural_network_early_stopping_patience"],
        "early_stopping_min_delta": config["neural_network_early_stopping_min_delta"],
        "lr_reduction_factor": config["neural_network_lr_reduction_factor"],
        "lr_reduction_patience": config["neural_network_lr_reduction_patience"],
        "minimum_learning_rate": config["neural_network_min_learning_rate"],
    }


def _fixed_xgboost_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": config["xgboost_estimators"],
        "max_depth": config["xgboost_max_depth"],
        "learning_rate": config["xgboost_learning_rate"],
        "subsample": config["xgboost_subsample"],
        "colsample_bytree": config["xgboost_colsample_bytree"],
        "min_child_weight": 1.0, "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
        "objective": "reg:squarederror", "n_jobs": config["xgboost_n_jobs"],
        "early_stopping_rounds": config["xgboost_early_stopping_rounds"],
    }


class _BlockedCandidateRuntime:
    dependency_versions = {"candidate_runtime": "unavailable"}

    def __init__(self, blocker: str) -> None:
        self.startup_blockers = (blocker,)

    def fit_fold(self, *args: Any, **kwargs: Any) -> CandidateFoldFit:
        raise RuntimeError("candidate runtime unavailable")

    def refit(self, *args: Any, **kwargs: Any) -> LockedFit:
        raise RuntimeError("candidate runtime unavailable")

    def load_locked(self, *args: Any, **kwargs: Any) -> Callable[..., Sequence[float]]:
        raise RuntimeError("candidate runtime unavailable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run private bounded MiniRes candidate tuning.")
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--volume-unit", default="mm3")
    parser.add_argument("--scope-confirmed", action="store_true")
    parser.add_argument("--seed", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        try:
            runtime: CandidateRuntime = TensorflowXGBoostCandidateRuntime()
        except ImportError:
            runtime = _BlockedCandidateRuntime("candidate_tuning_dependencies_required")
        except RuntimeError:
            runtime = _BlockedCandidateRuntime("candidate_tuning_runtime_unavailable")
        result = tune_candidates(
            args.records,
            EvaluationConfig(None, args.volume_unit,
                             True if args.scope_confirmed else None, seed=args.seed),
            runtime=runtime, output_root=args.output_root, limits=SearchLimits(seed=args.seed),
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, ValueError, TypeError):
        raise SystemExit("candidate_tuning_failed") from None
    print(json.dumps(result.to_dict(public=True), sort_keys=True, allow_nan=False))
    return 0


def _validate_limits(limits: SearchLimits) -> None:
    values = asdict(limits)
    if not isinstance(limits.seed, int) or isinstance(limits.seed, bool):
        raise ValueError("invalid_search_plan")
    if limits.second_seed is not None and (
        not isinstance(limits.second_seed, int) or isinstance(limits.second_seed, bool)
    ):
        raise ValueError("invalid_search_plan")
    expected = (6, 6, 3, 5, 20)
    actual = (limits.neural_network_trials, limits.xgboost_trials, limits.ensemble_trials,
              limits.second_seed_candidates, limits.maximum_candidate_runs)
    if actual != expected:
        raise ValueError("invalid_search_plan")
    if not isinstance(limits.maximum_elapsed_seconds, (int, float)) or isinstance(
        limits.maximum_elapsed_seconds, bool
    ) or not math.isfinite(limits.maximum_elapsed_seconds) or not (
        0 < limits.maximum_elapsed_seconds <= 7200.0
    ):
        raise ValueError("invalid_search_plan")
    if any(not isinstance(weight, (int, float)) or isinstance(weight, bool)
           or not math.isfinite(weight) or not 0 <= weight <= 1
           for weight in limits.ensemble_neural_network_weights):
        raise ValueError("invalid_search_plan")
    if tuple(sorted(set(limits.ensemble_neural_network_weights))) != limits.ensemble_neural_network_weights:
        raise ValueError("invalid_search_plan")
    if any(isinstance(value, float) and not math.isfinite(value)
           for value in values.values()):
        raise ValueError("invalid_search_plan")


if __name__ == "__main__":
    raise SystemExit(main())
