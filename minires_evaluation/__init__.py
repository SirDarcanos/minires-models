"""Evaluate physical sliced-resin-mass baselines from local records."""

from .evaluation import EvaluationConfig, PhysicalBaseline, evaluate_records
from .legacy import LegacyProvenance, LegacyReference, load_legacy_reference

__all__ = [
    "EvaluationConfig",
    "PhysicalBaseline",
    "LegacyProvenance",
    "LegacyReference",
    "evaluate_records",
    "load_legacy_reference",
]
