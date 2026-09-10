"""The public evaluation seam for local MiniRes baseline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import platform
from pathlib import Path
from typing import Any, Sequence

from .ingestion import CanonicalRow, Dataset, VOLUME_FACTORS, TRANSFORMATION_VERSION, fingerprint, load_records, normalize
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
                "baseline": "volume_density",
                "split_status": self.run_metadata.split_status,
                "python_version": self.run_metadata.python_version,
                "platform": self.run_metadata.platform,
                "transformation_version": self.run_metadata.transformation_version,
                "tolerance_g": self.run_metadata.tolerance_g,
                "scope_confirmed": self.run_metadata.scope_confirmed,
            },
        }
        if self.grouped_evaluation is not None:
            public_grouped_keys = ('source_count', 'sample_count', 'eligible_source_count',
                                   'unscored_input_count', 'pooled_weighting',
                                   'pooled_sample_denominator', 'source_balanced', 'limitations')
            summary['grouped_evaluation'] = (
                {key: self.grouped_evaluation[key] for key in public_grouped_keys}
                if public else self.grouped_evaluation)
        if public:
            return summary
        summary["run_metadata"] = asdict(self.run_metadata)
        summary["normalized_records"] = [asdict(record) for record in self.normalized_records]
        summary["predictions"] = [asdict(prediction) for prediction in self.predictions]
        summary["canonical_rows"] = [asdict(row) for row in self.canonical_rows]
        summary["reconciliations"] = list(self.reconciliations)
        return summary


def evaluate_records(
    records: Dataset,
    config: EvaluationConfig,
    baseline: PhysicalBaseline,
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
    _require_physical_baseline(baseline)
    if not isinstance(config.seed, int) or isinstance(config.seed, bool):
        from .ingestion import InputError
        raise InputError("invalid_seed")
    loaded, input_fingerprint = load_records(records)
    canonical_rows = normalize(loaded, config)
    normalized = [NormalizedRecord(row.features["volume_mm3"], row.sliced_resin_mass_g)
                  for row in canonical_rows
                  if row.outcome == "included" and row.features["volume_mm3"] is not None
                  and row.sliced_resin_mass_g is not None]
    reasons: dict[str, int] = {}
    for row in canonical_rows:
        for reason in row.reasons[:1]:
            reasons[reason] = reasons.get(reason, 0) + 1
    manifest = None
    if split_manifest is not None:
        from .splits import freeze_splits
        manifest = freeze_splits(canonical_rows, input_fingerprint, asdict(config), split_manifest)
    scoring_rows = [row for row in canonical_rows if row.outcome == 'included']
    if manifest is not None and manifest['status'] == 'blocked':
        scoring_rows = []
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
        baseline="volume_density",
        split_status=split_status,
        python_version=platform.python_version(),
        platform=platform.platform(),
        configuration_fingerprint=fingerprint(asdict(config)),
        transformation_version=TRANSFORMATION_VERSION,
        tolerance_g=config.tolerance_g if _is_finite_positive(config.tolerance_g) else None,
        scope_confirmed=config.scope_confirmed if isinstance(config.scope_confirmed, bool) else None,
        code_fingerprint=fingerprint({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}),
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
        grouped_evaluation=_grouped_report(manifest, scoring_rows, predictions, config.tolerance_g)
            if manifest is not None else None,
    )
    if output_dir is not None:
        from .artifacts import write_private
        write_private(result, output_dir)
    return result


def _grouped_report(manifest: dict[str, Any], rows: Sequence[CanonicalRow],
                    predictions: Sequence[Prediction], tolerance: float) -> dict[str, Any]:
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
    balanced: dict[str, Any] = {'weighting': 'each_source_equal_then_each_sample_within_source_equal',
                                'source_denominator': count, 'sample_count': len(predictions)}
    for key in ('mae_g', 'signed_error_g', 'within_tolerance_fraction', 'underestimation_fraction'):
        balanced[key] = math.fsum(summary[key] / count for summary in summaries) if count else None
    balanced['rmse_g'] = math.hypot(*(summary['rmse_g'] / math.sqrt(count)
                                                    for summary in summaries)) if count else None
    under_fraction = balanced['underestimation_fraction']
    balanced['mean_underestimation_g'] = (
        math.fsum((summary['mean_underestimation_g'] or 0) * summary['underestimation_fraction'] / count
                  for summary in summaries) / under_fraction if under_fraction else None)
    for key in ('absolute_error_bins', 'volume_bins'):
        balanced[key] = [{'label': item['label'],
                          'fraction': math.fsum(summary[key][i]['count'] / summary['sample_count'] / count
                                                for summary in summaries)}
                         for i, item in enumerate(summaries[0][key])] if count else []
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
