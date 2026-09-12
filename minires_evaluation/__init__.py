"""Evaluate physical sliced-resin-mass baselines from local records."""

from .evaluation import EvaluationConfig, PhysicalBaseline, evaluate_records
from .legacy import LegacyProvenance, LegacyReference, load_legacy_reference
from .learned import LearnedBaseline, LearnedBaselineConfig
from .model_definitions import (
    ModelKind, ModelRuntime, ModelSpecification, TrainingData, ValidationData,
    candidate_model_specification, combine_ensemble_predictions,
    ensemble_model_specification,
    fixed_model_specification,
)
from .preparation import PreparationResult, prepare_private_dataset
from .partitioning import PartitionResult, partition_private_dataset
from .stl_preparation import (
    ProcessResult,
    ProcessRunner,
    StlPreparationResult,
    SubprocessRunner,
    prepare_stl,
)

__all__ = [
    "EvaluationConfig",
    "PhysicalBaseline",
    "LegacyProvenance",
    "LegacyReference",
    "LearnedBaseline",
    "LearnedBaselineConfig",
    "ModelKind",
    "ModelRuntime",
    "ModelSpecification",
    "TrainingData",
    "ValidationData",
    "candidate_model_specification",
    "combine_ensemble_predictions",
    "ensemble_model_specification",
    "fixed_model_specification",
    "evaluate_records",
    "load_legacy_reference",
    "PreparationResult",
    "prepare_private_dataset",
    "PartitionResult",
    "partition_private_dataset",
    "ProcessResult",
    "ProcessRunner",
    "StlPreparationResult",
    "SubprocessRunner",
    "prepare_stl",
]
