"""Locked-candidate assessment on untouched anonymous-source evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from .evaluation import EvaluationConfig
from .ingestion import (
    CanonicalRow, Dataset, InputError, MalformedLocalRecord, load_records, normalize,
)
from .learned import _matrix, _predict
from .legacy import LegacyProvenance, LegacyReference, load_legacy_reference
from .private_io import write_private_json
from .tuning import (
    CandidateRuntime, LockedCandidate, TensorflowXGBoostCandidateRuntime,
    load_locked_candidate, verify_locked_candidate_files,
)


ASSESSMENT_VERSION = "minires-locked-assessment-v2"


@dataclass(frozen=True)
class FinalAssessmentConfig:
    seed: int = 1729
    bootstrap_replicates: int = 10_000
    minimum_sources: int = 3
    minimum_records_per_source: int = 200
    mae_noninferiority_margin: float = 0.02
    within_2g_noninferiority_margin: float = -0.01
    pooled_tail_limit: float = 0.01
    source_balanced_tail_limit: float = 0.01
    per_source_tail_limit: float = 0.02


@dataclass(frozen=True)
class AssessmentResult:
    status: str
    blockers: tuple[str, ...]
    promoted: bool
    row_accounting: dict[str, Any]
    observed: dict[str, Any]
    confidence_analysis: dict[str, Any]
    gates: dict[str, bool]
    source_reports: tuple[dict[str, Any], ...]
    predictions: tuple[dict[str, Any], ...]
    classification: str = "limited_unseen_source_evidence_internal_advisory"

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": ASSESSMENT_VERSION,
            "status": self.status,
            "blockers": list(self.blockers),
            "promoted": self.promoted,
            "row_accounting": {
                key: value for key, value in self.row_accounting.items()
                if not public or key not in {"source_counts"}
            },
            "observed": self.observed,
            "confidence_analysis": self.confidence_analysis,
            "gates": self.gates,
            "classification": self.classification,
            "use": "internal_advisory_with_human_review_of_every_estimate",
            "limitations": [
                "conditional_on_observed_anonymous_source_groups",
                "not_population_wide_performance",
                "no_prediction_interval",
                "not_actual_shop_consumption_or_pricing_accuracy",
                "not_operational_allowance",
            ],
        }
        if not public:
            result["source_reports"] = list(self.source_reports)
            result["predictions"] = list(self.predictions)
        return result


def assess_locked_candidate(
    records: Dataset,
    evaluation_config: EvaluationConfig,
    locked: LockedCandidate,
    legacy: LegacyReference,
    *,
    output_root: str | Path,
    runtime: CandidateRuntime,
    config: FinalAssessmentConfig = FinalAssessmentConfig(),
    supported_slicing_configuration: Mapping[str, Any] | None = None,
) -> AssessmentResult:
    """Verify a lock before loading final records, then run paired assessment."""
    output = Path(output_root)
    if "private" not in output.resolve().parts:
        raise InputError("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None
    lock_blockers = _verify_lock(locked, runtime.dependency_versions)
    if lock_blockers:
        result = _blocked(lock_blockers)
        _write_assessment(output, result)
        return result
    try:
        candidate_predictor = runtime.load_locked(
            locked.candidate, locked.directory, locked.contract
        )
    except Exception:
        result = _blocked(("locked_candidate_load_failed",))
        _write_assessment(output, result)
        return result
    if not _valid_config(config):
        result = _blocked(("invalid_assessment_plan",))
        _write_assessment(output, result)
        return result
    if legacy.blockers or legacy.neural_network is None or legacy.xgboost is None:
        result = _blocked(tuple(sorted(set(legacy.blockers) | {"legacy_reference_unavailable"})))
        _write_assessment(output, result)
        return result

    try:
        loaded, _ = load_records(records)
        if any(isinstance(record, MalformedLocalRecord) for record in loaded):
            raise InputError("malformed_local_record")
        rows = normalize(loaded, evaluation_config, contract="legacy")
    except (InputError, OSError, TypeError, ValueError):
        result = _blocked(("final_evidence_unavailable_or_malformed",))
        _write_assessment(output, result)
        return result
    expected_conditions = (
        dict(supported_slicing_configuration)
        if supported_slicing_configuration is not None else None
    )
    accepted = [
        row for row in rows
        if row.outcome == "included"
        and row.metadata.get("slicing_conditions")
        and (
            expected_conditions is None
            or row.metadata.get("slicing_conditions") == expected_conditions
        )
    ]
    condition_review = [
        row for row in rows
        if row.outcome == "included" and not row.metadata.get("slicing_conditions")
    ]
    configuration_review = [
        row for row in rows
        if row.outcome == "included"
        and row.metadata.get("slicing_conditions")
        and expected_conditions is not None
        and row.metadata.get("slicing_conditions") != expected_conditions
    ]
    source_counts: dict[str, int] = {}
    for row in accepted:
        source = row.metadata.get("anonymous_source_group")
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
    reasons: dict[str, int] = {}
    for row in rows:
        for reason in row.reasons:
            reasons[reason] = reasons.get(reason, 0) + 1
    if condition_review:
        reasons["final_slicing_conditions_required"] = len(condition_review)
    if configuration_review:
        reasons["unsupported_final_slicing_configuration"] = len(configuration_review)
    row_accounting = {
        "input_count": len(rows),
        "accepted_count": len(accepted),
        "needs_review_count": sum(row.outcome == "needs_review" for row in rows)
                              + len(condition_review) + len(configuration_review),
        "excluded_count": sum(row.outcome == "excluded" for row in rows),
        "source_count": len(source_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "reasons": dict(sorted(reasons.items())),
    }
    coverage_blockers = []
    if len(source_counts) < config.minimum_sources:
        coverage_blockers.append("insufficient_final_source_groups")
    if source_counts and any(count < config.minimum_records_per_source
                             for count in source_counts.values()):
        coverage_blockers.append("insufficient_final_source_records")
    if any(row.metadata.get("anonymous_source_group") is None for row in accepted):
        coverage_blockers.append("unresolved_final_source_group")
    if any(row.metadata.get("miniature_family") is None for row in accepted):
        coverage_blockers.append("unresolved_final_miniature_family")
    coverage_blockers.extend(_final_grouping_blockers(accepted))
    development_sources = set(locked.contract.get("development_source_groups", ()))
    if development_sources.intersection(source_counts):
        coverage_blockers.append("final_source_used_in_candidate_development")
    if condition_review:
        coverage_blockers.append("final_scope_evidence_incomplete")
    if configuration_review:
        coverage_blockers.append("unsupported_final_slicing_configuration")
    if len(accepted) != len(rows):
        coverage_blockers.append("final_row_accounting_incomplete")
    if coverage_blockers:
        result = _blocked(tuple(sorted(set(coverage_blockers))), row_accounting)
        _write_assessment(output, result)
        return result

    try:
        matrix, actual = _matrix(accepted)
        candidate = _predict(candidate_predictor, matrix)
        neural = _predict(legacy.neural_network, matrix)
        xgboost = _predict(legacy.xgboost, matrix)
        legacy_values = tuple(
            legacy.neural_network_weight * left + (1 - legacy.neural_network_weight) * right
            for left, right in zip(neural, xgboost)
        )
        if len(candidate) != len(legacy_values) or len(candidate) != len(accepted):
            raise ValueError("row mismatch")
    except Exception:
        result = _blocked(("paired_prediction_row_mismatch_or_failure",), row_accounting)
        _write_assessment(output, result)
        return result

    paired = tuple({
        "row_index": row.row_index,
        "source": row.metadata["anonymous_source_group"],
        "family": row.metadata["miniature_family"],
        "actual": target,
        "candidate": candidate_value,
        "legacy": legacy_value,
    } for row, target, candidate_value, legacy_value
       in zip(accepted, actual, candidate, legacy_values))
    observed, source_reports = _observed(paired)
    confidence = _clustered_bootstrap(paired, config)
    gates = evaluate_promotion_gates(observed, confidence, config)
    promoted = bool(gates) and all(gates.values())
    blockers = () if promoted else ("promotion_gates_not_met",)
    result = AssessmentResult(
        "promoted" if promoted else "not_promoted", blockers, promoted,
        row_accounting, observed, confidence, gates, source_reports, paired,
    )
    _write_assessment(output, result)
    return result


def evaluate_promotion_gates(
    observed: Mapping[str, Any], confidence: Mapping[str, Any],
    config: FinalAssessmentConfig = FinalAssessmentConfig(),
) -> dict[str, bool]:
    """Apply inclusive non-inferiority boundaries and observed absolute tails."""
    values = {
        "pooled_mae_noninferior": (confidence.get("pooled_mae_relative_regression_upper_95"),
                                   config.mae_noninferiority_margin, "upper"),
        "source_balanced_mae_noninferior": (
            confidence.get("source_balanced_mae_relative_regression_upper_95"),
            config.mae_noninferiority_margin, "upper"),
        "pooled_within_2g_noninferior": (
            confidence.get("pooled_within_2g_difference_lower_95"),
            config.within_2g_noninferiority_margin, "lower"),
        "source_balanced_within_2g_noninferior": (
            confidence.get("source_balanced_within_2g_difference_lower_95"),
            config.within_2g_noninferiority_margin, "lower"),
        "pooled_tail": (observed.get("pooled_above_5g_fraction"), config.pooled_tail_limit, "upper"),
        "source_balanced_tail": (observed.get("source_balanced_above_5g_fraction"),
                                 config.source_balanced_tail_limit, "upper"),
        "every_source_tail": (observed.get("maximum_source_above_5g_fraction"),
                              config.per_source_tail_limit, "upper"),
    }
    return {
        name: bool(isinstance(value, (int, float)) and math.isfinite(value)
                   and (value <= boundary if direction == "upper" else value >= boundary))
        for name, (value, boundary, direction) in values.items()
    }


def _observed(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source"]), []).append(row)
    reports = []
    for source, items in sorted(by_source.items()):
        metrics: dict[str, Any] = _paired_metrics(items)
        errors = [abs(float(item["candidate"]) - float(item["actual"])) for item in items]
        metrics["tail_confidence_interval_95"] = _wilson_interval(
            sum(error > 5.0 for error in errors), len(errors)
        )
        reports.append({"source": source, "sample_count": len(items), **metrics})
    pooled = _paired_metrics(rows)
    balanced_keys = ("candidate_mae_g", "legacy_mae_g", "candidate_within_2g_fraction",
                     "legacy_within_2g_fraction", "candidate_above_5g_fraction")
    balanced = {key: math.fsum(float(report[key]) for report in reports) / len(reports)
                for key in balanced_keys}
    pooled_tail_count = sum(abs(float(row["candidate"]) - float(row["actual"])) > 5.0
                            for row in rows)
    observed = {
        "sample_count": len(rows),
        "source_count": len(reports),
        "pooled_candidate_mae_g": pooled["candidate_mae_g"],
        "pooled_legacy_mae_g": pooled["legacy_mae_g"],
        "source_balanced_candidate_mae_g": balanced["candidate_mae_g"],
        "source_balanced_legacy_mae_g": balanced["legacy_mae_g"],
        "pooled_candidate_within_2g_fraction": pooled["candidate_within_2g_fraction"],
        "pooled_legacy_within_2g_fraction": pooled["legacy_within_2g_fraction"],
        "source_balanced_candidate_within_2g_fraction": balanced["candidate_within_2g_fraction"],
        "source_balanced_legacy_within_2g_fraction": balanced["legacy_within_2g_fraction"],
        "pooled_above_5g_fraction": pooled["candidate_above_5g_fraction"],
        "source_balanced_above_5g_fraction": balanced["candidate_above_5g_fraction"],
        "maximum_source_above_5g_fraction": max(
            float(report["candidate_above_5g_fraction"]) for report in reports
        ),
        "pooled_tail_confidence_interval_95": _wilson_interval(pooled_tail_count, len(rows)),
    }
    return observed, tuple(reports)


def _paired_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    count = len(rows)
    candidate_errors = [abs(float(row["candidate"]) - float(row["actual"])) for row in rows]
    legacy_errors = [abs(float(row["legacy"]) - float(row["actual"])) for row in rows]
    return {
        "candidate_mae_g": math.fsum(candidate_errors) / count,
        "legacy_mae_g": math.fsum(legacy_errors) / count,
        "candidate_within_2g_fraction": sum(value <= 2.0 for value in candidate_errors) / count,
        "legacy_within_2g_fraction": sum(value <= 2.0 for value in legacy_errors) / count,
        "candidate_above_5g_fraction": sum(value > 5.0 for value in candidate_errors) / count,
    }


def _clustered_bootstrap(rows: Sequence[Mapping[str, Any]],
                         config: FinalAssessmentConfig) -> dict[str, Any]:
    by_source_family: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        by_source_family.setdefault(str(row["source"]), {}).setdefault(
            str(row["family"]), []).append(row)
    rng = random.Random(config.seed)
    distributions: dict[str, list[float]] = {key: [] for key in (
        "pooled_mae", "balanced_mae", "pooled_within", "balanced_within",
        "pooled_tail", "balanced_tail",
    )}
    for _ in range(config.bootstrap_replicates):
        sample: list[Mapping[str, Any]] = []
        sampled_sources: list[list[Mapping[str, Any]]] = []
        for families in by_source_family.values():
            names = tuple(families)
            source_sample = [row for _ in names for row in families[rng.choice(names)]]
            sampled_sources.append(source_sample)
            sample.extend(source_sample)
        pooled = _bootstrap_metrics(sample)
        source_stats = [_bootstrap_metrics(items) for items in sampled_sources]
        balanced_candidate_mae = math.fsum(item[0] for item in source_stats) / len(source_stats)
        balanced_legacy_mae = math.fsum(item[1] for item in source_stats) / len(source_stats)
        distributions["pooled_mae"].append(
            pooled[0] / pooled[1] - 1.0 if pooled[1] > 0 else math.nan
        )
        distributions["balanced_mae"].append(
            balanced_candidate_mae / balanced_legacy_mae - 1.0
            if balanced_legacy_mae > 0 else math.nan
        )
        distributions["pooled_within"].append(pooled[2] - pooled[3])
        distributions["balanced_within"].append(
            math.fsum(item[2] for item in source_stats) / len(source_stats)
            - math.fsum(item[3] for item in source_stats) / len(source_stats)
        )
        distributions["pooled_tail"].append(pooled[4])
        distributions["balanced_tail"].append(
            math.fsum(item[4] for item in source_stats) / len(source_stats)
        )
    return {
        "method": "deterministic_family_clustered_bootstrap_within_observed_source",
        "replicates": config.bootstrap_replicates,
        "seed": config.seed,
        "confidence_level": 0.95,
        "interval_sidedness": "one_sided",
        "source_balanced_weighting": "each_observed_source_equal_in_every_replicate",
        "pooled_mae_relative_regression_upper_95": _quantile(distributions["pooled_mae"], 0.95),
        "source_balanced_mae_relative_regression_upper_95": _quantile(
            distributions["balanced_mae"], 0.95),
        "pooled_within_2g_difference_lower_95": _quantile(distributions["pooled_within"], 0.05),
        "source_balanced_within_2g_difference_lower_95": _quantile(
            distributions["balanced_within"], 0.05),
        "pooled_tail_confidence_interval_95": [
            _quantile(distributions["pooled_tail"], 0.025),
            _quantile(distributions["pooled_tail"], 0.975),
        ],
        "source_balanced_tail_confidence_interval_95": [
            _quantile(distributions["balanced_tail"], 0.025),
            _quantile(distributions["balanced_tail"], 0.975),
        ],
        "tail_intervals_are_reported_not_gated": True,
    }


def _bootstrap_metrics(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float, float, float, float]:
    metrics = _paired_metrics(rows)
    return (
        metrics["candidate_mae_g"], metrics["legacy_mae_g"],
        metrics["candidate_within_2g_fraction"], metrics["legacy_within_2g_fraction"],
        metrics["candidate_above_5g_fraction"],
    )


def _quantile(values: Sequence[float], probability: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if len(finite) != len(values) or not finite:
        return None
    index = max(0, min(len(finite) - 1, math.ceil(probability * len(finite)) - 1))
    return finite[index]


def _wilson_interval(successes: int, count: int) -> list[float]:
    if count <= 0:
        return [math.nan, math.nan]
    z = NormalDist().inv_cdf(0.975)
    fraction = successes / count
    denominator = 1 + z * z / count
    center = (fraction + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _final_grouping_blockers(rows: Sequence[CanonicalRow]) -> tuple[str, ...]:
    parent = {row.row_index: row.row_index for row in rows}

    def root(index: int) -> int:
        while parent[index] != index:
            index = parent[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for row in rows:
        for key in ("miniature_family", "duplicate_group", "geometry_fingerprint",
                    "record_identity", "location_evidence"):
            token = row.metadata.get(key)
            if token:
                evidence = (key, token)
                if evidence in seen:
                    left, right = root(row.row_index), root(seen[evidence])
                    parent[max(left, right)] = min(left, right)
                seen[evidence] = row.row_index
    sources: dict[int, set[str]] = {}
    for row in rows:
        source = row.metadata.get("anonymous_source_group")
        if source:
            sources.setdefault(root(row.row_index), set()).add(source)
    blockers = set()
    if any(len(values) > 1 for values in sources.values()):
        blockers.add("conflicting_final_source_evidence")
    origins: dict[str, set[str]] = {}
    aliases: dict[str, set[str]] = {}
    for row in rows:
        origin = row.metadata.get("source_identity_evidence")
        alias = row.metadata.get("anonymous_source_group")
        if origin and alias:
            origins.setdefault(origin, set()).add(alias)
            aliases.setdefault(alias, set()).add(origin)
    if any(len(values) > 1 for values in (*origins.values(), *aliases.values())):
        blockers.add("conflicting_final_source_evidence")
    return tuple(sorted(blockers))


def _verify_lock(locked: LockedCandidate, dependencies: Mapping[str, str]) -> tuple[str, ...]:
    blockers, _, _ = verify_locked_candidate_files(
        locked.directory, dependencies,
        expected_manifest=locked.manifest,
        expected_contract=locked.contract,
    )
    return blockers


def _valid_config(config: FinalAssessmentConfig) -> bool:
    return (
        isinstance(config.seed, int) and not isinstance(config.seed, bool)
        and config.bootstrap_replicates == 10_000
        and config.minimum_sources == 3
        and config.minimum_records_per_source == 200
        and config.mae_noninferiority_margin == 0.02
        and config.within_2g_noninferiority_margin == -0.01
        and config.pooled_tail_limit == 0.01
        and config.source_balanced_tail_limit == 0.01
        and config.per_source_tail_limit == 0.02
    )


def _blocked(blockers: Sequence[str], row_accounting: Mapping[str, Any] | None = None
             ) -> AssessmentResult:
    return AssessmentResult(
        "blocked", tuple(blockers), False,
        dict(row_accounting or {"input_count": 0, "accepted_count": 0,
                                "needs_review_count": 0, "excluded_count": 0,
                                "source_count": 0, "reasons": {}}),
        {}, {}, {}, (), (),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess one locked MiniRes candidate privately.")
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--locked-candidate", required=True, type=Path)
    parser.add_argument("--legacy-artifacts", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--volume-unit", default="mm3")
    parser.add_argument("--scope-confirmed", action="store_true")
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        try:
            runtime = TensorflowXGBoostCandidateRuntime()
        except ImportError:
            return _write_cli_startup_blocker(
                args.output_root, "candidate_assessment_dependencies_required"
            )
        except RuntimeError:
            return _write_cli_startup_blocker(
                args.output_root, "candidate_assessment_runtime_unavailable"
            )
        try:
            locked = load_locked_candidate(args.locked_candidate, runtime)
        except InputError as error:
            return _write_cli_startup_blocker(args.output_root, str(error))
        legacy = load_legacy_reference(
            args.legacy_artifacts, provenance=LegacyProvenance.unknown()
        )
        result = assess_locked_candidate(
            args.records,
            EvaluationConfig(None, args.volume_unit,
                             True if args.scope_confirmed else None),
            locked, legacy, output_root=args.output_root, runtime=runtime,
            config=FinalAssessmentConfig(seed=args.bootstrap_seed),
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, ValueError, TypeError):
        raise SystemExit("candidate_assessment_failed") from None
    print(json.dumps(result.to_dict(public=True), sort_keys=True, allow_nan=False))
    return 0


def _write_cli_startup_blocker(output: Path, blocker: str) -> int:
    if "private" not in output.resolve().parts:
        raise SystemExit("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        result = _blocked((blocker,))
        _write_assessment(output, result)
    except OSError:
        raise SystemExit("private_output_directory_unavailable") from None
    print(json.dumps(result.to_dict(public=True), sort_keys=True, allow_nan=False))
    return 0


def _write_assessment(output: Path, result: AssessmentResult) -> None:
    write_private_json(output / "assessment.json", result.to_dict())
    public = result.to_dict(public=True)
    public["publication_status"] = "draft_not_approved"
    write_private_json(output / "public-summary-draft.json", public)
    from .evidence import review_public_summary
    write_private_json(output / "public-summary-review.json", review_public_summary(public))
    inventory = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"
    }
    write_private_json(output / "manifest.json", {
        "version": ASSESSMENT_VERSION,
        "artifacts": inventory,
        "publication_performed": False,
    })


if __name__ == "__main__":
    raise SystemExit(main())
