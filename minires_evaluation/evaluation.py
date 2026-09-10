"""The public evaluation seam for local MiniRes baseline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import platform
from typing import Any, Mapping, Sequence


SUPPORTED_VOLUME_UNITS = {"mm3"}
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

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        """Serialize a result; public output exposes only an allowlisted summary."""
        summary = {
            "status": self.status,
            "split_status": self.split_status,
            "blockers": list(self.blockers),
            "metrics": asdict(self.metrics),
            "data_quality": asdict(self.data_quality),
            "run_metadata": {
                "seed": self.run_metadata.seed,
                "volume_unit": self.run_metadata.volume_unit,
                "resin_density_g_per_ml": self.run_metadata.resin_density_g_per_ml,
                "baseline": self.run_metadata.baseline,
                "split_status": self.run_metadata.split_status,
                "python_version": self.run_metadata.python_version,
                "platform": self.run_metadata.platform,
            },
        }
        if public:
            return summary
        summary["run_metadata"]["input_fingerprint"] = self.run_metadata.input_fingerprint
        summary["normalized_records"] = [asdict(record) for record in self.normalized_records]
        summary["predictions"] = [asdict(prediction) for prediction in self.predictions]
        return summary


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    config: EvaluationConfig,
    baseline: PhysicalBaseline,
) -> EvaluationResult:
    """Evaluate local records at the public evaluation seam.

    The physical baseline only supports cubic-millimetre volume and an explicit
    resin density in grams per millilitre. It intentionally creates no split:
    grouped holdout evaluation is a later capability.
    """
    _require_physical_baseline(baseline)
    blockers = _configuration_blockers(config)
    normalized, reasons = _normalize(records, config, blocked=bool(blockers))
    predictions = () if blockers else _predict(normalized, config.resin_density_g_per_ml)
    metrics = _metrics(predictions, normalized, config.tolerance_g)
    accepted_count = len(normalized)
    needs_review_count = sum(reasons.values())
    status = _status(blockers, accepted_count, needs_review_count)
    metadata = RunMetadata(
        input_fingerprint=_fingerprint(normalized),
        seed=config.seed,
        volume_unit=config.volume_unit,
        resin_density_g_per_ml=config.resin_density_g_per_ml,
        baseline=baseline.name,
        split_status="not_applicable",
        python_version=platform.python_version(),
        platform=platform.platform(),
    )
    return EvaluationResult(
        status=status,
        split_status="not_applicable",
        blockers=tuple(blockers),
        normalized_records=tuple(normalized),
        predictions=tuple(predictions),
        metrics=metrics,
        data_quality=DataQualitySummary(
            input_count=len(records),
            accepted_count=accepted_count,
            needs_review_count=needs_review_count,
            reasons=dict(sorted(reasons.items())),
        ),
        run_metadata=metadata,
    )


def _require_physical_baseline(baseline: PhysicalBaseline) -> None:
    if not isinstance(baseline, PhysicalBaseline):
        raise TypeError("baseline must be PhysicalBaseline")


def _configuration_blockers(config: EvaluationConfig) -> list[str]:
    blockers: list[str] = []
    if config.resin_density_g_per_ml is None:
        blockers.append("resin_density_required")
    elif not _is_finite_positive(config.resin_density_g_per_ml):
        blockers.append("invalid_resin_density")
    if config.volume_unit not in SUPPORTED_VOLUME_UNITS:
        blockers.append("unsupported_volume_unit")
    if not _is_finite_positive(config.tolerance_g):
        blockers.append("invalid_tolerance")
    return blockers


def _normalize(
    records: Sequence[Mapping[str, Any]], config: EvaluationConfig, *, blocked: bool
) -> tuple[list[NormalizedRecord], dict[str, int]]:
    normalized: list[NormalizedRecord] = []
    reasons: dict[str, int] = {}
    for record in records:
        reason = _record_reason(record, config, blocked)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        normalized.append(
            NormalizedRecord(
                volume_mm3=float(record["volume"]),
                sliced_resin_mass_g=float(_target_sliced_resin_mass(record)),
            )
        )
    return normalized, reasons


def _record_reason(
    record: Mapping[str, Any], config: EvaluationConfig, blocked: bool
) -> str | None:
    if not _is_finite_positive(record.get("volume")):
        return "invalid_volume"
    target = _target_sliced_resin_mass(record)
    if target is None:
        return "missing_target_sliced_resin_mass"
    if not _is_finite_nonnegative(target):
        return "invalid_target_sliced_resin_mass"
    if blocked:
        return "evaluation_configuration_blocked"
    if config.scope_confirmed is not True:
        return "scope_confirmation_required"
    return None


def _target_sliced_resin_mass(record: Mapping[str, Any]) -> Any:
    """Read the canonical target, retaining legacy CSV compatibility."""
    if "sliced_resin_mass_g" in record:
        return record["sliced_resin_mass_g"]
    return record.get("weight")


def _predict(
    records: Sequence[NormalizedRecord], density_g_per_ml: float | None
) -> tuple[Prediction, ...]:
    assert density_g_per_ml is not None
    return tuple(
        Prediction(
            predicted_sliced_resin_mass_g=record.volume_mm3 / 1_000.0 * density_g_per_ml,
            actual_sliced_resin_mass_g=record.sliced_resin_mass_g,
        )
        for record in records
    )


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
        mae_g=sum(absolute_errors) / count,
        rmse_g=math.sqrt(sum(error**2 for error in errors) / count),
        signed_error_g=sum(errors) / count,
        within_tolerance_fraction=within_count / count,
        within_tolerance_percent=within_count / count * 100,
        underestimation_count=len(underestimations),
        underestimation_fraction=len(underestimations) / count,
        mean_underestimation_g=(sum(underestimations) / len(underestimations)) if underestimations else None,
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


def _fingerprint(records: Sequence[NormalizedRecord]) -> str:
    canonical = "\n".join(
        f"{record.volume_mm3:.17g},{record.sliced_resin_mass_g:.17g}" for record in records
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _is_finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _is_finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
