"""Auditable command-level candidate tuning and final-assessment workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import resource
import time
from typing import Any, Mapping, Sequence

from .assessment import (
    AssessmentResult,
    FinalAssessmentConfig,
    assess_locked_candidate,
)
from .baseline import EvaluationConfig
from .evidence import review_public_summary
from ..ingestion import InputError
from ..modeling.legacy import LegacyProvenance, LegacyReference, load_legacy_reference
from ..private_io import create_private_file, write_private_json
from ..modeling.tuning import (
    CandidateRuntime,
    LockedCandidate,
    SearchLimits,
    TensorflowXGBoostCandidateRuntime,
    TuningResult,
    load_locked_candidate,
    tune_candidates,
)


WORKFLOW_VERSION = "minires-end-to-end-tuning-v1"
_SUPPORTED_SLICING_FIELDS = {
    "layer_height_mm", "exposure_seconds", "bottom_exposure_seconds",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_bytes(path: Path, content: bytes) -> None:
    with create_private_file(path) as stream:
        stream.write(content)


def _inventory(path: Path) -> dict[str, Any]:
    try:
        files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
        if not files or any(not item.is_file() for item in files):
            return {"status": "unavailable", "files": {}}
        return {
            "status": "recorded",
            "files": {
                f"file_{index}": {
                    "size_bytes": item.stat().st_size,
                    "sha256": sha256(item.read_bytes()).hexdigest(),
                }
                for index, item in enumerate(files)
            },
        }
    except OSError:
        return {"status": "unavailable", "files": {}}


def _input_snapshot(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {name: _inventory(path) for name, path in sorted(paths.items())}


def _artifact_checksums(root: Path) -> dict[str, str]:
    excluded = {
        "evidence-index.json", "public-summary-draft.json",
        "public-summary-review.json", "manifest.json",
    }
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def _load_slicing_configuration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        raise InputError("invalid_slicing_configuration") from None
    if not isinstance(value, dict):
        raise InputError("invalid_slicing_configuration")
    conditions = value.get("slicing_conditions")
    if not isinstance(conditions, dict) or not conditions or not set(conditions) <= _SUPPORTED_SLICING_FIELDS:
        raise InputError("invalid_slicing_configuration")
    normalized: dict[str, float] = {}
    for key, raw in conditions.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
            raise InputError("invalid_slicing_configuration")
        normalized[key] = float(raw)
    volume_unit = value.get("volume_unit")
    if volume_unit not in {"mm3", "cm3", "ml"}:
        raise InputError("invalid_slicing_configuration")
    return value, normalized


def _resources(started: float, cpu_started: float) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "elapsed_seconds": max(0.0, time.perf_counter() - started),
        "process_cpu_seconds": max(0.0, time.process_time() - cpu_started),
        "process_peak_rss_platform_units": usage.ru_maxrss,
        "peak_rss_scope": "workflow_process_high_water_mark",
    }


def _public_summary(
    *,
    status: str,
    blockers: Sequence[str],
    mode: str,
    tuning: TuningResult | None,
    assessment: AssessmentResult | None,
    resources: Mapping[str, Any],
) -> dict[str, Any]:
    tuning_summary: dict[str, Any] = {
        "status": "not_run",
        "candidate_selected": False,
        "run_count": 0,
        "allocation": {},
        "resource_use": {},
    }
    if tuning is not None:
        tuning_summary = {
            "status": tuning.status,
            "candidate_selected": tuning.locked_candidate is not None,
            "run_count": tuning.run_count,
            "allocation": dict(tuning.allocation),
            "resource_use": dict(tuning.resource_use),
        }
    assessment_summary: dict[str, Any] = {
        "status": "not_run",
        "promoted": False,
        "metrics": {},
        "confidence_metrics": {},
        "gate_outcomes": {},
        "row_accounting": {},
    }
    if assessment is not None:
        public_assessment = assessment.to_dict(public=True)
        assessment_summary = {
            "status": assessment.status,
            "promoted": assessment.promoted,
            "metrics": dict(assessment.observed),
            "confidence_metrics": dict(assessment.confidence_analysis),
            "gate_outcomes": dict(assessment.gates),
            "row_accounting": public_assessment["row_accounting"],
        }
    return {
        "version": WORKFLOW_VERSION,
        "publication_status": "draft_not_approved",
        "status": status,
        "blockers": sorted(set(blockers)),
        "mode": mode,
        "tuning": tuning_summary,
        "assessment": assessment_summary,
        "resource_summary": dict(resources),
        "limitations": [
            "manual_content_review_required_before_any_publication",
            "no_publication_performed",
            "internal_advisory_with_human_review_of_every_estimate",
            "conditional_on_observed_anonymous_source_groups",
            "not_population_wide_performance",
            "no_prediction_interval",
            "not_actual_shop_consumption_or_pricing_accuracy",
            "issue_11_supplies_at_most_one_of_three_required_final_sources",
            "at_least_two_additional_qualifying_untouched_sources_are_required",
        ],
    }


def run_end_to_end_workflow(
    *,
    final_records: Path,
    legacy_artifacts: Path,
    slicing_configuration: Path,
    output_root: Path,
    evaluation_config: EvaluationConfig,
    bootstrap_seed: int,
    runtime: CandidateRuntime,
    development_records: Path | None = None,
    locked_candidate: Path | None = None,
    limits: SearchLimits | None = None,
    legacy_reference: LegacyReference | None = None,
) -> dict[str, Any]:
    """Run tuning plus assessment, or assessment-only, into one create-only package."""
    output = Path(output_root)
    if "private" not in output.resolve().parts:
        raise InputError("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None

    started = time.perf_counter()
    cpu_started = time.process_time()
    mode = "assessment_only" if locked_candidate is not None else "tune_and_assess"
    blockers: list[str] = []
    tuning: TuningResult | None = None
    assessment: AssessmentResult | None = None
    locked: LockedCandidate | None = None
    slicing_contract: dict[str, Any] = {}
    supported_conditions: dict[str, Any] = {}

    fixed_inputs = {
        "legacy_artifacts": Path(legacy_artifacts),
        "slicing_configuration": Path(slicing_configuration),
    }
    if development_records is not None:
        fixed_inputs["development_records"] = Path(development_records)
    if locked_candidate is not None:
        fixed_inputs["locked_candidate"] = Path(locked_candidate)
    before = _input_snapshot(fixed_inputs)
    final_before: dict[str, Any] = {"status": "not_accessed_before_lock", "files": {}}
    final_after: dict[str, Any] = {"status": "not_accessed_before_lock", "files": {}}
    assessment_resources: dict[str, Any] = {}
    tuning_started = False
    assessment_started = False
    assessment_started_at = 0.0
    assessment_cpu_started = 0.0
    interrupted = False

    try:
        slicing_contract, supported_conditions = _load_slicing_configuration(
            Path(slicing_configuration)
        )
        if slicing_contract.get("volume_unit") != evaluation_config.volume_unit:
            raise InputError("slicing_configuration_volume_unit_mismatch")
        if mode == "tune_and_assess":
            if development_records is None or limits is None:
                raise InputError("development_workflow_arguments_required")
            tuning_started = True
            tuning = tune_candidates(
                Path(development_records), evaluation_config,
                runtime=runtime, output_root=output / "tuning", limits=limits,
            )
            blockers.extend(tuning.blockers)
            locked = tuning.locked_candidate
            if locked is None:
                blockers.append("locked_candidate_not_created")
            else:
                try:
                    locked = load_locked_candidate(locked.directory, runtime)
                except InputError as error:
                    blockers.append(str(error))
                    locked = None
        else:
            assert locked_candidate is not None
            try:
                locked = load_locked_candidate(locked_candidate, runtime)
            except InputError as error:
                blockers.append(str(error))

        if locked is not None:
            final_before = _inventory(Path(final_records))
            assessment_started = True
            assessment_started_at = time.perf_counter()
            assessment_cpu_started = time.process_time()
            legacy = legacy_reference or load_legacy_reference(
                legacy_artifacts, provenance=LegacyProvenance.unknown()
            )
            assessment = assess_locked_candidate(
                Path(final_records), evaluation_config, locked, legacy,
                output_root=output / "assessment", runtime=runtime,
                config=FinalAssessmentConfig(seed=bootstrap_seed),
                supported_slicing_configuration=supported_conditions,
            )
            assessment_resources = _resources(
                assessment_started_at, assessment_cpu_started
            )
            blockers.extend(assessment.blockers)
            final_after = _inventory(Path(final_records))
    except KeyboardInterrupt:
        interrupted = True
        blockers.append("workflow_interrupted")
    except InputError as error:
        blockers.append(str(error))
    except (OSError, TypeError, ValueError):
        blockers.append("end_to_end_workflow_failed")

    if final_before.get("status") == "recorded" and final_after.get("status") != "recorded":
        final_after = _inventory(Path(final_records))
    if assessment_started and not assessment_resources:
        assessment_resources = _resources(assessment_started_at, assessment_cpu_started)
    after = _input_snapshot(fixed_inputs)
    input_integrity = {
        "unchanged": before == after and final_before == final_after,
        "before": {**before, "final_records": final_before},
        "after": {**after, "final_records": final_after},
        "write_scope": "new_private_output_root_only",
    }
    if assessment is not None and assessment.promoted:
        status = "completed"
        promotion_decision = "promoted_for_internal_advisory_use"
    elif assessment is not None and assessment.status == "not_promoted":
        status = "completed_not_promoted"
        promotion_decision = "not_promoted"
    elif assessment is not None:
        status = "completed_with_blockers"
        promotion_decision = "blocked"
    else:
        status = "blocked"
        promotion_decision = "not_assessed"

    resources = _resources(started, cpu_started)
    public = _public_summary(
        status=status, blockers=blockers, mode=mode, tuning=tuning,
        assessment=assessment, resources=resources,
    )
    if tuning is None and tuning_started:
        public["tuning"]["status"] = "interrupted" if interrupted else "failed"
        public["tuning"]["run_count"] = None
    if assessment is None and assessment_started:
        public["assessment"]["status"] = "interrupted" if interrupted else "failed"
    review = review_public_summary(public)
    public_bytes = _json_bytes(public)
    review_bytes = _json_bytes(review)
    completed_artifacts = _artifact_checksums(output)
    completed_artifacts["public-summary-draft.json"] = sha256(public_bytes).hexdigest()
    completed_artifacts["public-summary-review.json"] = sha256(review_bytes).hexdigest()

    if tuning is not None:
        identities = {
            "code": tuning.plan.code_fingerprint,
            "configuration": tuning.plan.configuration_fingerprint,
            "code_configuration": tuning.plan.code_configuration_fingerprint,
            "raw_input": tuning.plan.input_fingerprint,
            "normalized_input": tuning.plan.normalized_input_fingerprint,
            "source_allocation": tuning.plan.source_allocation_fingerprint,
            "development_split": (
                tuning.locked_candidate.contract["development_evidence"][
                    "development_split_fingerprint"
                ] if tuning.locked_candidate is not None else None
            ),
        }
    elif locked is not None:
        development_evidence = locked.contract.get("development_evidence", {})
        identities = {
            "code": locked.contract.get("code_fingerprint"),
            "configuration": None,
            "code_configuration": None,
            "raw_input": development_evidence.get("input_fingerprint"),
            "normalized_input": development_evidence.get("normalized_input_fingerprint"),
            "source_allocation": development_evidence.get("source_allocation_fingerprint"),
            "development_split": development_evidence.get("development_split_fingerprint"),
        }
    else:
        identities = {}

    evidence = {
        "version": WORKFLOW_VERSION,
        "classification": "private_end_to_end_tuning_evidence",
        "status": status,
        "blockers": sorted(set(blockers)),
        "mode": mode,
        "invocation": {
            "development_records": str(development_records) if development_records else None,
            "final_records": str(final_records),
            "legacy_artifacts": str(legacy_artifacts),
            "locked_candidate": str(locked_candidate) if locked_candidate else None,
            "slicing_configuration": str(slicing_configuration),
            "output_root": str(output_root),
        },
        "configuration": {
            "evaluation": asdict(evaluation_config),
            "search_limits": asdict(limits) if limits is not None else None,
            "bootstrap_seed": bootstrap_seed,
            "supported_slicing_configuration": slicing_contract,
        },
        "identities": identities,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": dict(sorted(runtime.dependency_versions.items())),
        },
        "input_integrity": input_integrity,
        "phases": {
            "tuning": {
                "status": (
                    tuning.status if tuning is not None else
                    "interrupted" if tuning_started and interrupted else
                    "failed" if tuning_started else "not_run"
                ),
                "completed_candidate_runs": (
                    tuning.run_count if tuning is not None else None if tuning_started else 0
                ),
                "candidate_locking_completed": bool(
                    tuning is not None and tuning.locked_candidate is not None
                ),
                "evidence_directory": "tuning" if tuning is not None else None,
            },
            "assessment": {
                "status": (
                    assessment.status if assessment is not None else
                    "interrupted" if assessment_started and interrupted else
                    "failed" if assessment_started else "not_run"
                ),
                "promotion_decision": promotion_decision,
                "evidence_directory": "assessment" if assessment is not None else None,
                "resource_use": assessment_resources,
            },
        },
        "resource_use": resources,
        "artifact_checksums": completed_artifacts,
        "publication": {
            "draft_created": True,
            "automated_screening_status": review["status"],
            "manual_content_review_required": True,
            "publication_performed": False,
        },
        "interpretation_limits": public["limitations"],
    }
    write_private_json(output / "evidence-index.json", evidence)
    _write_bytes(output / "public-summary-draft.json", public_bytes)
    _write_bytes(output / "public-summary-review.json", review_bytes)
    root_inventory = {
        str(path.relative_to(output)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != output / "manifest.json"
    }
    write_private_json(output / "manifest.json", {
        "version": WORKFLOW_VERSION,
        "create_only": True,
        "artifacts": root_inventory,
        "manual_content_review_required": True,
        "publication_performed": False,
    })
    return evidence


class _UnavailableRuntime:
    dependency_versions = {"candidate_runtime": "unavailable"}
    startup_blockers = ("candidate_tuning_dependencies_required",)

    def fit_fold(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("candidate runtime unavailable")

    def refit(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("candidate runtime unavailable")

    def load_locked(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("candidate runtime unavailable")

    def serialize_fold(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("candidate runtime unavailable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run auditable MiniRes tuning and final assessment privately."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development-records", type=Path)
    mode.add_argument("--assessment-only", action="store_true")
    parser.add_argument("--locked-candidate", type=Path)
    parser.add_argument("--final-records", required=True, type=Path)
    parser.add_argument("--legacy-artifacts", required=True, type=Path)
    parser.add_argument("--slicing-configuration", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--density-g-per-ml", type=float)
    parser.add_argument("--volume-unit", required=True)
    parser.add_argument("--scope-confirmed", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--second-seed", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--maximum-candidate-runs", type=int, default=20)
    parser.add_argument("--maximum-elapsed-seconds", type=float, default=7200.0)
    parser.add_argument("--neural-network-trials", type=int, default=6)
    parser.add_argument("--xgboost-trials", type=int, default=6)
    parser.add_argument("--ensemble-trials", type=int, default=3)
    parser.add_argument("--second-seed-candidates", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.assessment_only and args.locked_candidate is None:
        raise SystemExit("locked_candidate_required_for_assessment_only")
    if not args.assessment_only and args.locked_candidate is not None:
        raise SystemExit("locked_candidate_only_valid_for_assessment_only")
    try:
        try:
            runtime: CandidateRuntime = TensorflowXGBoostCandidateRuntime()
        except (ImportError, RuntimeError):
            runtime = _UnavailableRuntime()
        limits = None if args.assessment_only else SearchLimits(
            seed=args.seed,
            second_seed=args.second_seed,
            maximum_candidate_runs=args.maximum_candidate_runs,
            maximum_elapsed_seconds=args.maximum_elapsed_seconds,
            neural_network_trials=args.neural_network_trials,
            xgboost_trials=args.xgboost_trials,
            ensemble_trials=args.ensemble_trials,
            second_seed_candidates=args.second_seed_candidates,
        )
        evidence = run_end_to_end_workflow(
            development_records=args.development_records,
            final_records=args.final_records,
            legacy_artifacts=args.legacy_artifacts,
            locked_candidate=args.locked_candidate,
            slicing_configuration=args.slicing_configuration,
            output_root=args.output_root,
            evaluation_config=EvaluationConfig(
                args.density_g_per_ml, args.volume_unit,
                True if args.scope_confirmed else None, seed=args.seed,
            ),
            bootstrap_seed=args.bootstrap_seed,
            limits=limits,
            runtime=runtime,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, TypeError, ValueError):
        raise SystemExit("end_to_end_workflow_failed") from None
    print(json.dumps({
        "status": evidence["status"],
        "blockers": evidence["blockers"],
        "publication_performed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
