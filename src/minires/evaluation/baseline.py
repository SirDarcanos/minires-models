"""The public evaluation seam for local MiniRes baseline experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import platform
from pathlib import Path
from typing import Any, Sequence, cast

from ..ingestion import CanonicalRow, Dataset, VOLUME_FACTORS, TRANSFORMATION_VERSION, fingerprint, load_records, normalize
from ..modeling.legacy import LegacyReference
from ..modeling.learned import LearnedBaseline, LearnedRun, fit_frozen_folds
from ..source_identity import code_fingerprint
from .reconciliation import reconcile


SUPPORTED_VOLUME_UNITS = set(VOLUME_FACTORS)
ABSOLUTE_ERROR_BIN_EDGES_G = (0.0, 2.0, 5.0)
VOLUME_BIN_EDGES_MM3 = (0.0, 1_000.0, 5_000.0, 20_000.0)


@dataclass(frozen=True)
class EvaluationConfig:
    """Explicit conditions needed to evaluate a physical baseline."""

    resin_density_g_per_ml: float | None
    volume_unit: str | None
    scope_confirmed: bool | None
    seed: int = 0
    tolerance_g: float = 2.0


@dataclass(frozen=True)
class PhysicalBaseline:
    """Estimate sliced resin mass from volume and an explicit resin density."""

    name: str = "volume_density"


@dataclass(frozen=True)
class NormalizedRecord:
    volume_mm3: float
    sliced_resin_mass_g: float


@dataclass(frozen=True)
class Prediction:
    predicted_sliced_resin_mass_g: float
    actual_sliced_resin_mass_g: float


@dataclass(frozen=True)
class BinCount:
    label: str
    count: int


@dataclass(frozen=True)
class MetricSummary:
    sample_count: int
    mae_g: float | None
    rmse_g: float | None
    signed_error_g: float | None
    within_tolerance_fraction: float | None
    within_tolerance_percent: float | None
    underestimation_count: int
    underestimation_fraction: float | None
    mean_underestimation_g: float | None
    absolute_error_bins: tuple[BinCount, ...]
    volume_bins: tuple[BinCount, ...]


@dataclass(frozen=True)
class DataQualitySummary:
    input_count: int
    accepted_count: int
    needs_review_count: int
    excluded_count: int
    reasons: dict[str, int]


@dataclass(frozen=True)
class RunMetadata:
    input_fingerprint: str
    seed: int
    volume_unit: str | None
    resin_density_g_per_ml: float | None
    baseline: str
    split_status: str
    python_version: str
    platform: str
    configuration_fingerprint: str
    transformation_version: str
    tolerance_g: float | None
    scope_confirmed: bool | None
    code_fingerprint: str


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    split_status: str
    blockers: tuple[str, ...]
    normalized_records: tuple[NormalizedRecord, ...]
    predictions: tuple[Prediction, ...]
    metrics: MetricSummary
    data_quality: DataQualitySummary
    run_metadata: RunMetadata
    canonical_rows: tuple[CanonicalRow, ...]
    reconciliations: tuple[dict[str, Any], ...]
    model_diagnostics: dict[str, MetricSummary]
    provenance_classification: str
    provenance_evidence: dict[str, Any] | None
    model_contract: dict[str, Any]
    grouped_evaluation: dict[str, Any] | None = None

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        """Serialize a result; public output exposes only an allowlisted summary."""
        summary: dict[str, Any] = {
            "status": self.status,
            "split_status": self.split_status,
            "blockers": list(self.blockers),
            "metrics": asdict(self.metrics),
            "data_quality": asdict(self.data_quality),
            "run_metadata": {
                "seed": self.run_metadata.seed,
                "volume_unit": self.run_metadata.volume_unit if self.run_metadata.volume_unit in SUPPORTED_VOLUME_UNITS else None,
                "resin_density_g_per_ml": self.run_metadata.resin_density_g_per_ml,
                "baseline": self.run_metadata.baseline,
                "split_status": self.run_metadata.split_status,
                "python_version": self.run_metadata.python_version,
                "platform": self.run_metadata.platform,
                "transformation_version": self.run_metadata.transformation_version,
                "tolerance_g": self.run_metadata.tolerance_g,
                "scope_confirmed": self.run_metadata.scope_confirmed,
            },
        }
        summary["provenance_classification"] = self.provenance_classification
        summary["model_contract"] = deepcopy(self.model_contract)
        if public:
            summary["model_contract"].get("run", {}).pop("split_fingerprint", None)
            for private_key in ("repository", "revision", "artifacts"):
                summary["model_contract"].pop(private_key, None)
        summary["model_diagnostics"] = {
            name: asdict(metrics) for name, metrics in self.model_diagnostics.items()
        }
        if self.grouped_evaluation is not None:
            public_grouped_keys = ('source_count', 'sample_count', 'eligible_source_count',
                                   'unscored_input_count', 'pooled_weighting',
                                   'pooled_sample_denominator', 'source_balanced',
                                   'component_source_balanced', 'limitations')
            summary['grouped_evaluation'] = (
                {key: self.grouped_evaluation[key] for key in public_grouped_keys
                 if key in self.grouped_evaluation}
                if public else self.grouped_evaluation)
        if public:
            return summary
        summary["provenance_evidence"] = self.provenance_evidence
        summary["run_metadata"] = asdict(self.run_metadata)
        summary["normalized_records"] = [asdict(record) for record in self.normalized_records]
        summary["predictions"] = [asdict(prediction) for prediction in self.predictions]
        summary["canonical_rows"] = [asdict(row) for row in self.canonical_rows]
        summary["reconciliations"] = list(self.reconciliations)
        return summary


def evaluate_records(
    records: Dataset,
    config: EvaluationConfig,
    baseline: PhysicalBaseline | LegacyReference | LearnedBaseline,
    *,
    reconcile_with: Sequence[Dataset] = (),
    output_dir: str | Path | None = None,
    split_manifest: str | Path | None = None,
) -> EvaluationResult:
    """Evaluate local records at the public evaluation seam.

    The physical baseline only supports cubic-millimetre volume and an explicit
    resin density in grams per millilitre. Supplying a private split manifest
    enables frozen source holdouts; otherwise this is an ungrouped diagnostic.
    """
    legacy_model = baseline if isinstance(baseline, LegacyReference) else None
    learned_model = baseline if isinstance(baseline, LearnedBaseline) else None
    is_legacy = legacy_model is not None
    is_learned = learned_model is not None
    if legacy_model is None and learned_model is None:
        assert isinstance(baseline, PhysicalBaseline)
        _require_physical_baseline(baseline)
    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        from ..ingestion import InputError
        raise InputError("invalid_seed")
    loaded, input_fingerprint = load_records(records)
    canonical_rows = normalize(loaded, config, contract="legacy" if is_legacy or is_learned else "canonical")
    normalized = [NormalizedRecord(row.features["volume_mm3"], row.sliced_resin_mass_g)
                  for row in canonical_rows
                  if row.outcome == "included" and row.features["volume_mm3"] is not None
                  and row.sliced_resin_mass_g is not None]
    reasons: dict[str, int] = {}
    for row in canonical_rows:
        for reason in row.reasons[:1]:
            reasons[reason] = reasons.get(reason, 0) + 1
    manifest = None
    run_configuration = asdict(config)
    if learned_model is not None:
        run_configuration["learned_baseline"] = asdict(learned_model.config)
    if split_manifest is not None:
        from .splits import freeze_splits
        # Allocation depends only on data eligibility and split controls. Model
        # configuration is recorded with the run, but must not create a
        # different holdout for each baseline.
        split_configuration = asdict(config)
        manifest = freeze_splits(
            canonical_rows, input_fingerprint, split_configuration, split_manifest
        )
    scoring_rows = [row for row in canonical_rows if row.outcome == 'included']
    if manifest is not None and manifest['status'] == 'blocked':
        scoring_rows = []
    model_diagnostics: dict[str, MetricSummary] = {}
    predictions: tuple[Prediction, ...]
    inference_blockers: tuple[str, ...]
    learned_run: LearnedRun | None = None
    if is_legacy:
        assert legacy_model is not None
        provenance_blockers = legacy_model.provenance.validate_sources(canonical_rows)
        if provenance_blockers:
            predictions, model_diagnostics, inference_blockers = (), {}, ()
        else:
            predictions, model_diagnostics, inference_blockers = _legacy_predictions(
                legacy_model, scoring_rows, config.tolerance_g
            )
        blockers = sorted(
            set(legacy_model.blockers) | set(provenance_blockers) | set(inference_blockers)
        )
    elif is_learned:
        assert learned_model is not None
        if manifest is None:
            predictions, blockers = (), ["learned_baseline_split_manifest_required"]
        else:
            artifact_root = Path(output_dir) / "fitted-folds" if output_dir is not None else None
            learned_run = fit_frozen_folds(learned_model, scoring_rows, manifest, artifact_root)
            predictions, model_diagnostics = _learned_predictions(learned_run, scoring_rows, config.tolerance_g)
            blockers = list(learned_run.blockers)
    else:
        predictions = tuple(Prediction(row.features["volume_mm3"] / 1000.0 * row.metadata["resin_density_g_per_ml"], row.sliced_resin_mass_g)
                            for row in scoring_rows
                            if row.features["volume_mm3"] is not None and row.sliced_resin_mass_g is not None)
        blockers = sorted({reason for row in canonical_rows for reason in row.reasons
                           if reason in {"resin_density_required", "invalid_resin_density", "unsupported_volume_unit", "invalid_tolerance"}}) if not predictions else []
    if manifest is not None:
        blockers = sorted(set(blockers) | set(manifest['blockers']))
    metrics = _metrics(predictions, normalized if predictions else (), config.tolerance_g)
    split_status = manifest['status'] if manifest is not None else 'not_applicable'
    accepted_count = len(normalized)
    needs_review_count = sum(row.outcome == "needs_review" for row in canonical_rows)
    excluded_count = sum(row.outcome == "excluded" for row in canonical_rows)
    status = _status(blockers, accepted_count, needs_review_count + excluded_count)
    metadata = RunMetadata(
        input_fingerprint=input_fingerprint,
        seed=config.seed,
        volume_unit=config.volume_unit if config.volume_unit in SUPPORTED_VOLUME_UNITS else None,
        resin_density_g_per_ml=config.resin_density_g_per_ml if _is_finite_positive(config.resin_density_g_per_ml) else None,
        baseline=(legacy_model.name if legacy_model is not None else
                  learned_model.name if learned_model is not None else "volume_density"),
        split_status=split_status,
        python_version=platform.python_version(),
        platform=platform.platform(),
        configuration_fingerprint=fingerprint(run_configuration),
        transformation_version=TRANSFORMATION_VERSION,
        tolerance_g=config.tolerance_g if _is_finite_positive(config.tolerance_g) else None,
        scope_confirmed=config.scope_confirmed if isinstance(config.scope_confirmed, bool) else None,
        code_fingerprint=code_fingerprint(),
    )
    reconciliations = []
    for dataset in reconcile_with:
        comparison, comparison_fingerprint = load_records(dataset)
        reconciliations.append(reconcile(canonical_rows, normalize(comparison, config), comparison_fingerprint))
    result = EvaluationResult(
        status=status,
        split_status=split_status,
        blockers=tuple(blockers),
        normalized_records=tuple(normalized),
        predictions=tuple(predictions),
        metrics=metrics,
        data_quality=DataQualitySummary(
            input_count=len(loaded),
            accepted_count=accepted_count,
            needs_review_count=needs_review_count,
            excluded_count=excluded_count,
            reasons=dict(sorted(reasons.items())),
        ),
        run_metadata=metadata,
        canonical_rows=tuple(canonical_rows),
        reconciliations=tuple(reconciliations),
        model_diagnostics=model_diagnostics,
        provenance_classification=(
            "legacy_reference_source_holdout_unverified"
            if legacy_model is not None
            and legacy_model.provenance.status == "source_held_out"
            and any(blocker.startswith("source_holdout_") for blocker in blockers)
            else legacy_model.provenance.classification if legacy_model is not None
            else "clean_unseen_source_evaluation" if learned_model is not None and not blockers
            else "not_applicable"
        ),
        provenance_evidence=(legacy_model.provenance.private_evidence if legacy_model is not None else None),
        model_contract=(legacy_model.contract if legacy_model is not None else
                        learned_run.contract if learned_run is not None else
                        learned_model.contract if learned_model is not None else {
            "classification": "physical_baseline",
            "name": "volume_density",
            "input": "volume_mm3",
            "density_unit": "g_per_ml",
            "output_unit": "g",
        }),
        grouped_evaluation=(_learned_grouped_report(manifest, learned_run, config.tolerance_g)
                            if manifest is not None and learned_run is not None else
                            _grouped_report(
                                manifest,
                                scoring_rows,
                                predictions,
                                config.tolerance_g,
                                predictions_blocked=bool(blockers),
                            )
                            if manifest is not None else None),
    )
    if output_dir is not None:
        from ..preparation.artifacts import write_private
        write_private(result, output_dir)
    return result


def _legacy_predictions(
    model: Any, rows: Sequence[CanonicalRow], tolerance: float
) -> tuple[tuple[Prediction, ...], dict[str, MetricSummary], tuple[str, ...]]:
    if model.blockers or model.neural_network is None or model.xgboost is None:
        return (), {}, ()
    if not rows:
        empty = _metrics((), (), tolerance)
        return (), {name: empty for name in ("neural_network", "xgboost", "ensemble")}, ()
    from ..modeling.legacy import prepare_canonical_legacy_features

    try:
        matrix = tuple(prepare_canonical_legacy_features(row) for row in rows)
        neural_values = tuple(float(value) for value in model.neural_network(matrix))
        xgboost_values = tuple(float(value) for value in model.xgboost(matrix))
        if len(neural_values) != len(rows) or len(xgboost_values) != len(rows):
            return (), {}, ("legacy_prediction_count_mismatch",)
        if not all(math.isfinite(value) for value in neural_values + xgboost_values):
            return (), {}, ("non_finite_legacy_prediction",)
        actual = tuple(row.sliced_resin_mass_g for row in rows)
        if any(value is None for value in actual):
            return (), {}, ("legacy_target_unavailable",)
        component_values = {
            "neural_network": neural_values,
            "xgboost": xgboost_values,
            "ensemble": tuple(
                model.neural_network_weight * neural + (1.0 - model.neural_network_weight) * xgboost
                for neural, xgboost in zip(neural_values, xgboost_values)
            ),
        }
        records = tuple(
            NormalizedRecord(
                cast(float, row.features["volume_mm3"]),
                cast(float, row.sliced_resin_mass_g),
            )
            for row in rows
        )
        component_predictions = {
            name: tuple(Prediction(value, cast(float, target)) for value, target in zip(values, actual))
            for name, values in component_values.items()
        }
        diagnostics = {
            name: _metrics(values, records, tolerance)
            for name, values in component_predictions.items()
        }
        return component_predictions["ensemble"], diagnostics, ()
    except Exception:
        # Third-party runtimes can include input values or local paths in errors.
        return (), {}, ("legacy_inference_failed",)


def _learned_predictions(
    run: LearnedRun, rows: Sequence[CanonicalRow], tolerance: float
) -> tuple[tuple[Prediction, ...], dict[str, MetricSummary]]:
    if run.blockers:
        return (), {}
    by_index = {row.row_index: row for row in rows}
    component_values: dict[str, list[Prediction]] = {
        "neural_network": [], "xgboost": [], "ensemble": []}
    records: list[NormalizedRecord] = []
    for fold in run.folds:
        for position, row_index in enumerate(fold.test_rows):
            row = by_index[row_index]
            target = fold.actual[position]
            records.append(NormalizedRecord(cast(float, row.features["volume_mm3"]), target))
            for name in component_values:
                value = cast(tuple[float, ...], getattr(fold, name))[position]
                component_values[name].append(Prediction(value, target))
    diagnostics = {name: _metrics(values, records, tolerance)
                   for name, values in component_values.items()}
    return tuple(component_values["ensemble"]), diagnostics


def _balanced_metrics(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(summaries)
    result: dict[str, Any] = {
        "weighting": "each_source_equal_then_each_sample_within_source_equal",
        "source_denominator": count,
        "sample_count": sum(summary["sample_count"] for summary in summaries),
    }
    for key in ("mae_g", "signed_error_g", "within_tolerance_fraction", "underestimation_fraction"):
        result[key] = math.fsum(summary[key] / count for summary in summaries) if count else None
    result["rmse_g"] = math.hypot(*(summary["rmse_g"] / math.sqrt(count)
                                    for summary in summaries)) if count else None
    under_fraction = result["underestimation_fraction"]
    result["mean_underestimation_g"] = (
        math.fsum((summary["mean_underestimation_g"] or 0) * summary["underestimation_fraction"] / count
                  for summary in summaries) / under_fraction if under_fraction else None)
    for key in ("absolute_error_bins", "volume_bins"):
        result[key] = [{"label": item["label"],
                        "fraction": math.fsum(summary[key][i]["count"] / summary["sample_count"] / count
                                              for summary in summaries)}
                       for i, item in enumerate(summaries[0][key])] if count else []
    return result


def _learned_grouped_report(manifest: dict[str, Any], run: LearnedRun,
                            tolerance: float) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for fold in run.folds:
        predictions = {
            name: tuple(Prediction(value, actual) for value, actual in zip(getattr(fold, name), fold.actual))
            for name in ("neural_network", "xgboost", "ensemble")
        }
        fold_records = tuple(NormalizedRecord(volume, actual)
                             for volume, actual in zip(fold.volume_mm3, fold.actual))
        reports.append({
            "source": fold.source,
            "metrics": asdict(_metrics(predictions["ensemble"], fold_records, tolerance)),
            "component_metrics": {name: asdict(_metrics(values, fold_records, tolerance))
                                  for name, values in predictions.items()},
            "predictions": {name: [asdict(value) for value in values]
                            for name, values in predictions.items()},
            "selected_neural_network_weight": fold.neural_network_weight,
            "train_count": len(fold.fit_audit["train_rows"]),
            "validation_count": len(fold.fit_audit["validation_rows"]),
            "test_count": len(fold.test_rows),
            "fit_audit": fold.fit_audit,
            "fit_metadata": fold.fit_metadata,
        })
    count = len(reports)
    summaries = [report["metrics"] for report in reports]
    sample_count = sum(report["test_count"] for report in reports)
    balanced = _balanced_metrics(summaries)
    return {
        "manifest": manifest, "source_reports": reports,
        "source_count": count, "sample_count": sample_count,
        "eligible_source_count": manifest["eligible_source_count"],
        "unscored_input_count": len(manifest["unscored_rows"]) + (len(manifest["included_rows"]) if not count else 0),
        "pooled_weighting": "each_held_out_sample_equal_once",
        "pooled_sample_denominator": sample_count,
        "source_balanced": balanced,
        "component_source_balanced": {
            name: _balanced_metrics([report["component_metrics"][name] for report in reports])
            for name in ("neural_network", "xgboost", "ensemble")
        },
        "limitations": [item for item in manifest["limitations"]
                        if item != "no_fitting_or_model_selection"],
    }


def _grouped_report(
    manifest: dict[str, Any],
    rows: Sequence[CanonicalRow],
    predictions: Sequence[Prediction],
    tolerance: float,
    *,
    predictions_blocked: bool = False,
) -> dict[str, Any]:
    if predictions_blocked and not predictions:
        return {
            'manifest': manifest,
            'source_reports': [],
            'source_count': 0,
            'sample_count': 0,
            'eligible_source_count': manifest['eligible_source_count'],
            'unscored_input_count': len(manifest['unscored_rows']) + len(manifest['included_rows']),
            'pooled_weighting': 'each_held_out_sample_equal_once',
            'pooled_sample_denominator': 0,
            'source_balanced': _balanced_metrics([]),
            'limitations': [*manifest['limitations'], 'predictions_unavailable'],
        }
    if len(predictions) != len(rows):
        raise ValueError("prediction_count_mismatch")
    by_index = {row.row_index: (row, prediction) for row, prediction in zip(rows, predictions)}
    reports = []
    for fold in manifest['folds']:
        pairs = [by_index[index] for index in fold['test']]
        metrics = _metrics([pair[1] for pair in pairs],
                           [NormalizedRecord(pair[0].features['volume_mm3'], pair[1].actual_sliced_resin_mass_g)
                            for pair in pairs if pair[0].features['volume_mm3'] is not None], tolerance)
        reports.append({'source': fold['source'], 'metrics': asdict(metrics),
                        'train_count': len(fold['train']), 'validation_count': len(fold['validation']),
                        'test_count': len(fold['test'])})
    count = len(reports)
    summaries = [report['metrics'] for report in reports]
    balanced = _balanced_metrics(summaries)
    return {'manifest': manifest, 'source_reports': reports,
            'source_count': count, 'sample_count': len(predictions),
            'eligible_source_count': manifest['eligible_source_count'],
            'unscored_input_count': len(manifest['unscored_rows']) + (len(manifest['included_rows']) if not count else 0),
            'pooled_weighting': 'each_held_out_sample_equal_once',
            'pooled_sample_denominator': len(predictions),
            'source_balanced': balanced, 'limitations': manifest['limitations']}


def _require_physical_baseline(baseline: PhysicalBaseline) -> None:
    if not isinstance(baseline, PhysicalBaseline):
        raise TypeError("baseline must be PhysicalBaseline")


def _metrics(
    predictions: Sequence[Prediction],
    records: Sequence[NormalizedRecord],
    tolerance_g: float,
) -> MetricSummary:
    if not predictions:
        return MetricSummary(
            sample_count=0,
            mae_g=None,
            rmse_g=None,
            signed_error_g=None,
            within_tolerance_fraction=None,
            within_tolerance_percent=None,
            underestimation_count=0,
            underestimation_fraction=None,
            mean_underestimation_g=None,
            absolute_error_bins=_bin_counts((), ABSOLUTE_ERROR_BIN_EDGES_G, "g"),
            volume_bins=_bin_counts((), VOLUME_BIN_EDGES_MM3, "mm3"),
        )
    errors = tuple(
        prediction.predicted_sliced_resin_mass_g - prediction.actual_sliced_resin_mass_g
        for prediction in predictions
    )
    absolute_errors = tuple(abs(error) for error in errors)
    underestimations = tuple(-error for error in errors if error < 0)
    count = len(errors)
    within_count = sum(error <= tolerance_g for error in absolute_errors)
    return MetricSummary(
        sample_count=count,
        mae_g=math.fsum(error / count for error in absolute_errors),
        rmse_g=math.hypot(*(error / math.sqrt(count) for error in errors)),
        signed_error_g=math.fsum(error / count for error in errors),
        within_tolerance_fraction=within_count / count,
        within_tolerance_percent=within_count / count * 100,
        underestimation_count=len(underestimations),
        underestimation_fraction=len(underestimations) / count,
        mean_underestimation_g=math.fsum(error / len(underestimations) for error in underestimations) if underestimations else None,
        absolute_error_bins=_bin_counts(absolute_errors, ABSOLUTE_ERROR_BIN_EDGES_G, "g"),
        volume_bins=_bin_counts(
            tuple(record.volume_mm3 for record in records), VOLUME_BIN_EDGES_MM3, "mm3"
        ),
    )


def _bin_counts(values: Sequence[float], edges: tuple[float, ...], unit: str) -> tuple[BinCount, ...]:
    bins: list[BinCount] = []
    for index, lower in enumerate(edges):
        upper = edges[index + 1] if index + 1 < len(edges) else None
        if upper is None:
            label = f"({lower:.1f}, inf)"
            count = sum(value > lower for value in values)
        elif index == 0:
            label = f"[{lower:.1f}, {upper:.1f}]"
            count = sum(lower <= value <= upper for value in values)
        else:
            label = f"({lower:.1f}, {upper:.1f}]"
            count = sum(lower < value <= upper for value in values)
        bins.append(BinCount(label=label, count=count))
    return tuple(bins)


def _status(blockers: Sequence[str], accepted_count: int, review_count: int) -> str:
    if blockers:
        return "blocked"
    if accepted_count == 0 and review_count:
        return "needs_review"
    if review_count:
        return "completed_with_review"
    return "completed"


def _is_finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
