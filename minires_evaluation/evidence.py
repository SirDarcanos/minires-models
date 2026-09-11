"""Private evidence assembly and public-summary review for baseline runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import resource
import time
from typing import Any, Mapping, Sequence

from .evaluation import EvaluationConfig, EvaluationResult, PhysicalBaseline, evaluate_records
from .ingestion import InputError
from .learned import LearnedBaseline, enable_synchronous_dataset_execution
from .legacy import LegacyProvenance, load_legacy_reference
from .private_io import write_private_json


PUBLIC_PRIVATE_KEYS = {
    "canonical_rows",
    "configuration_fingerprint",
    "error",
    "input_fingerprint",
    "local_path",
    "normalized_records",
    "predictions",
    "provenance_evidence",
    "reconciliations",
    "source_mapping",
    "source_reports",
    "split_fingerprint",
}
ROW_LEVEL_KEYS = {"canonical_rows", "normalized_records", "predictions", "source_reports"}
PATH_MARKERS = ("/Users/", "/home/", "private/", "\\Users\\")
PRIVATE_KEY_MARKERS = (
    "artist", "fingerprint", "checksum", "source_mapping", "source_identity",
    "anonymous_source_group", "miniature_family", "row_index", "local_path",
)
SMALL_SOURCE_COUNT_THRESHOLD = 5
DEPENDENCIES = ("pyarrow", "numpy", "keras", "tensorflow", "xgboost")


def assess_repeatability(
    first: EvaluationResult,
    second: EvaluationResult,
    *,
    exact_predictions: bool = False,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
) -> dict[str, Any]:
    """Compare repeated results without weakening split or row accounting checks."""
    numerical = first.model_contract.get("numerical_reproducibility", {})
    absolute = 0.0 if exact_predictions else (
        float(absolute_tolerance)
        if absolute_tolerance is not None
        else float(numerical.get("absolute_tolerance", 1e-6))
    )
    relative = 0.0 if exact_predictions else (
        float(relative_tolerance)
        if relative_tolerance is not None
        else float(numerical.get("relative_tolerance", 1e-6))
    )
    first_manifest = _split_manifest(first)
    second_manifest = _split_manifest(second)
    split_exact = first_manifest == second_manifest
    accounting_exact = first.data_quality == second.data_quality
    metadata_exact = all(
        getattr(first.run_metadata, key) == getattr(second.run_metadata, key)
        for key in (
            "input_fingerprint",
            "configuration_fingerprint",
            "transformation_version",
            "code_fingerprint",
        )
    )
    numerical_result_observed = bool(first.predictions or second.predictions)
    predictions_match = len(first.predictions) == len(second.predictions) and all(
        math.isclose(
            left.predicted_sliced_resin_mass_g,
            right.predicted_sliced_resin_mass_g,
            abs_tol=absolute,
            rel_tol=relative,
        )
        and left.actual_sliced_resin_mass_g == right.actual_sliced_resin_mass_g
        for left, right in zip(first.predictions, second.predictions)
    )
    blockers_exact = first.blockers == second.blockers
    repeatable = all(
        (split_exact, accounting_exact, metadata_exact, predictions_match, blockers_exact)
    )
    return {
        "status": (
            "not_repeatable"
            if not repeatable
            else "repeatable"
            if numerical_result_observed
            else "structurally_repeatable_no_numerical_result"
        ),
        "split_exact": split_exact,
        "row_accounting_exact": accounting_exact,
        "run_identity_exact": metadata_exact,
        "blockers_exact": blockers_exact,
        "predictions_within_tolerance": predictions_match if numerical_result_observed else None,
        "prediction_comparison_status": (
            "compared" if numerical_result_observed else "not_observed_no_predictions"
        ),
        "prediction_count": len(first.predictions),
        "absolute_tolerance": absolute,
        "relative_tolerance": relative,
    }


def build_public_summary_draft(
    results: Mapping[str, EvaluationResult],
) -> dict[str, Any]:
    """Build a separate, aggregate-only draft; this does not publish or approve it."""
    source_counts = [
        grouped.get("source_count", 0)
        for result in results.values()
        if (grouped := result.to_dict(public=True).get("grouped_evaluation"))
    ]
    weakest_source_count = min(source_counts, default=0)
    return {
        "publication_status": "draft_not_approved",
        "results": {
            name: result.to_dict(public=True) for name, result in sorted(results.items())
        },
        "limitations": {
            "unseen_source_claim_strength": (
                "limited: source coverage is small and does not establish population-wide performance"
                if weakest_source_count < SMALL_SOURCE_COUNT_THRESHOLD
                else "bounded to the anonymous source groups represented by the frozen evaluation"
            ),
            "scope_outcomes": (
                "invalid inputs, unvalidated categories, and missing scope confirmation remain distinct outcomes"
            ),
            "uncertainty": "unvalidated; no prediction interval is claimed",
            "release_thresholds": "not agreed; this draft records evidence only",
            "compute_budget": "not agreed; no candidate tuning was performed",
        },
    }


def review_public_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the publication checklist to a concrete allowlisted draft."""
    findings: set[str] = set()

    def visit(value: Any, key: str | None = None) -> None:
        normalized_key = key.lower() if isinstance(key, str) else ""
        if (
            key in PUBLIC_PRIVATE_KEYS
            or any(marker in normalized_key for marker in PRIVATE_KEY_MARKERS)
        ):
            findings.add("identifying_or_private_key")
        if key in ROW_LEVEL_KEYS or normalized_key in {"rows", "row", "records"}:
            findings.add("row_level_output")
        if key == "error" or normalized_key.endswith("_error"):
            findings.add("raw_error_detail")
        if isinstance(value, str) and any(marker in value for marker in PATH_MARKERS):
            findings.add("local_path")
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)

    visit(summary)
    return {
        "status": "automated_screening_passed" if not findings else "rejected",
        "checks": [
            "identifying_metadata_and_source_mappings",
            "local_paths_and_fingerprints",
            "row_level_and_notebook_output",
            "raw_error_detail",
        ],
        "findings": sorted(findings),
        "screening_scope": "key_and_path_markers_only; manual content review still required",
        "manual_approval_required": True,
        "publication_performed": False,
    }


def _split_manifest(result: EvaluationResult) -> Any:
    grouped = result.grouped_evaluation
    return grouped.get("manifest") if grouped is not None else None


def _file_inventory(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(paths):
        if path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
        else:
            files = [path]
        for file_index, item in enumerate(files):
            data = item.read_bytes()
            inventory[f"input_{index}_{file_index}"] = {
                "size_bytes": len(data),
                "sha256": sha256(data).hexdigest(),
            }
    return inventory


def _environment() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for dependency in DEPENDENCIES:
        try:
            versions[dependency] = metadata.version(dependency)
        except metadata.PackageNotFoundError:
            versions[dependency] = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": versions,
    }


def _measure(call: Any) -> tuple[EvaluationResult, dict[str, Any]]:
    started = time.perf_counter()
    cpu_started = time.process_time()
    result = call()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return result, {
        "elapsed_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu_started,
        "process_peak_rss_platform_units": usage.ru_maxrss,
        "peak_rss_scope": "evidence_process_high_water_mark",
    }


def produce_evidence_package(
    *,
    records: Path,
    reconciliations: Sequence[Path],
    output_root: Path,
    split_manifest: Path,
    legacy_artifacts: Path,
    config: EvaluationConfig,
    legacy_provenance: LegacyProvenance | None = None,
    verification: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Run every supported baseline twice and write a private evidence index."""
    if "private" not in output_root.resolve().parts:
        raise InputError("private_output_directory_required")
    if output_root.exists():
        raise InputError("private_output_directory_unavailable")
    output_root.mkdir(parents=True, mode=0o700)
    # Configure tf.data before the preceding legacy baseline can initialize
    # TensorFlow; enabling this only when the learned adapter is constructed is
    # too late in a combined evidence process.
    enable_synchronous_dataset_execution()
    original_paths = [records, *reconciliations, legacy_artifacts]
    before = _file_inventory(original_paths)
    baseline_factories = {
        "physical": lambda: PhysicalBaseline(),
        "legacy_reference": lambda: load_legacy_reference(
            legacy_artifacts,
            provenance=legacy_provenance or LegacyProvenance.unknown(),
        ),
        "clean_fixed_configuration": lambda: LearnedBaseline(),
    }
    first_results: dict[str, EvaluationResult] = {}
    run_records: dict[str, Any] = {}
    for name, factory in baseline_factories.items():
        baseline_split_manifest = split_manifest.with_name(
            f"{split_manifest.stem}-{name}{split_manifest.suffix or '.json'}"
        )
        pair: list[EvaluationResult] = []
        resources: list[dict[str, Any]] = []
        for repetition in (1, 2):
            result, measured = _measure(
                lambda repetition=repetition, factory=factory, name=name: evaluate_records(
                    records,
                    config,
                    factory(),
                    reconcile_with=reconciliations,
                    output_dir=output_root / f"{name}-run-{repetition}",
                    split_manifest=baseline_split_manifest,
                )
            )
            pair.append(result)
            resources.append(measured)
        first_results[name] = pair[0]
        run_records[name] = {
            "status": pair[0].status,
            "blockers": list(pair[0].blockers),
            "classification": pair[0].provenance_classification,
            "data_quality": asdict(pair[0].data_quality),
            "resource_use": resources,
            "split_manifest_checksum": sha256(baseline_split_manifest.read_bytes()).hexdigest(),
            "repeatability": assess_repeatability(
                pair[0], pair[1], exact_predictions=name != "clean_fixed_configuration"
            ),
        }
    after = _file_inventory(original_paths)
    public_draft = build_public_summary_draft(first_results)
    public_review = review_public_summary(public_draft)
    write_private_json(output_root / "public-summary-draft.json", public_draft)
    write_private_json(output_root / "public-summary-review.json", public_review)
    evidence = {
        "classification": "private_baseline_evidence",
        "invocation": {
            "module": "minires_evaluation.evidence",
            "records": str(records),
            "reconciliations": [str(path) for path in reconciliations],
            "output_root": str(output_root),
            "split_manifest_template": str(split_manifest),
            "legacy_artifacts": str(legacy_artifacts),
        },
        "configuration": asdict(config),
        "environment": _environment(),
        "verification": [dict(item) for item in verification],
        "input_integrity": {
            "unchanged": before == after,
            "before": before,
            "after": after,
        },
        "runs": run_records,
        "conclusion": {
            "threshold_decision": "not_made",
            "future_compute_budget": "not_agreed",
            "candidate_tuning_performed": False,
            "replacement_weights_selected": False,
            "required_evidence_blocked": any(result.blockers for result in first_results.values()),
        },
        "artifact_checksums": {
            str(path.relative_to(output_root)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(output_root.rglob("*"))
            if path.is_file()
        },
    }
    write_private_json(output_root / "evidence.json", evidence)
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Produce a private, repeated baseline evidence package.")
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--reconcile", action="append", default=[], type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--legacy-artifacts", required=True, type=Path)
    parser.add_argument("--density-g-per-ml", type=float)
    parser.add_argument("--volume-unit", default="mm3")
    parser.add_argument("--scope-confirmed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verification-record",
        type=Path,
        help="Private JSON array of previously completed synthetic checks and outcomes",
    )
    return parser


def _load_verification(path: Path | None) -> Sequence[Mapping[str, str]]:
    if path is None:
        return ()
    try:
        loaded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        raise InputError("invalid_verification_record") from None
    if not isinstance(loaded, list) or not all(
        isinstance(item, dict)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in item.items())
        for item in loaded
    ):
        raise InputError("invalid_verification_record")
    return loaded


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evidence = produce_evidence_package(
            records=args.records,
            reconciliations=args.reconcile,
            output_root=args.output_root,
            split_manifest=args.split_manifest,
            legacy_artifacts=args.legacy_artifacts,
            config=EvaluationConfig(
                args.density_g_per_ml,
                args.volume_unit,
                True if args.scope_confirmed else None,
                seed=args.seed,
            ),
            verification=_load_verification(args.verification_record),
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, ValueError, TypeError):
        raise SystemExit("local_evidence_failed") from None
    print(json.dumps({
        "status": "completed_with_blockers" if evidence["conclusion"]["required_evidence_blocked"] else "completed",
        "evidence_path": str(args.output_root / "evidence.json"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
