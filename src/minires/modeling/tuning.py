"""Deterministic, leakage-resistant candidate tuning at one orchestration seam."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
from pathlib import Path
import platform
import random
import resource
import statistics
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..evaluation import EvaluationConfig
from ..ingestion import (
    CanonicalRow, Dataset, InputError, TRANSFORMATION_VERSION, fingerprint,
    load_records, normalize,
)
from .learned import LearnedBaselineConfig, _matrix, _predict
from .legacy import LEGACY_FEATURES
from .definitions import (
    FittedModel, ModelKind, ModelRuntime, ModelSpecification,
    TensorflowXGBoostBackend, TrainingData,
    ValidationData, candidate_model_specification, combine_ensemble_predictions,
    ensemble_model_specification,
    fixed_model_specification,
)
from ..private_io import create_private_file, write_private_json
from ..source_identity import code_fingerprint
from ..evaluation.splits import ALLOCATION_VERSION, freeze_splits


TUNING_VERSION = "minires-candidate-tuning-v5"
EXPLICIT_DEVELOPMENT_EVIDENCE_VERSION = "source-grouped-validation-v1"
DEVELOPMENT_ONLY_STAGES = (
    "fitting", "preprocessing", "early_stopping", "ensemble_selection",
    "threshold_selection", "candidate_locking",
)
POOLED_ABOVE_5G_FRACTION_MAXIMUM = 0.01
SOURCE_BALANCED_ABOVE_5G_FRACTION_MAXIMUM = 0.01
PER_SOURCE_ABOVE_5G_FRACTION_MAXIMUM = 0.02
PER_SOURCE_MINIMUM_ACCEPTED_RECORDS = 200
TAIL_ALIGNED_VALIDATION_SELECTION = "serious_error_gates_then_ranking_v1"

# This representation is deliberately source-neutral. Every value is derived from
# canonical geometry; source, identity, family, linkage, and legacy proxy metadata
# are unavailable to the feature builder.
GEOMETRY_REGIME_FEATURES = (
    "volume_mm3", "surface_area_mm2", "bounding_box_short_mm",
    "bounding_box_middle_mm", "bounding_box_long_mm",
    "bounding_box_volume_mm3", "euler_number", "log1p_volume_mm3",
    "log1p_surface_area_mm2", "log1p_bounding_box_volume_mm3",
    "log1p_bounding_box_short_mm", "log1p_bounding_box_middle_mm",
    "log1p_bounding_box_long_mm", "log_volume_to_bounding_box_volume_ratio",
    "log_surface_to_volume_ratio_per_mm", "log_bounding_box_long_to_short_ratio",
)
GEOMETRY_REGIME_INPUTS = (
    "volume_mm3", "surface_area_mm2", "bounding_box_x_mm",
    "bounding_box_y_mm", "bounding_box_z_mm", "bounding_box_volume_mm3",
    "euler_number",
)
GEOMETRY_REGIME_TRANSFORMATION_VERSION = "minires-geometry-regime-features-v1"
LEGACY_GEOMETRY_AUGMENTATION_FEATURES = LEGACY_FEATURES + (
    "mesh_volume_to_bounding_box_volume_ratio",
    "surface_area_to_bounding_box_volume_ratio_per_mm",
    "log1p_volume_squared",
    "bounding_box_long_to_short_ratio",
)
LEGACY_GEOMETRY_AUGMENTATION_TRANSFORMATION_VERSION = (
    "minires-legacy-geometry-augmentation-v1"
)
CROSS_FITTED_GEOMETRY_GATE_TRANSFORMATION_VERSION = (
    "minires-cross-fitted-geometry-gate-v1"
)
CROSS_FITTED_GEOMETRY_GATE_FEATURES = LEGACY_FEATURES + GEOMETRY_REGIME_FEATURES
CROSS_FIT_FOLDS = 5
GEOMETRY_GATE_RIDGE_PENALTIES = (0.01, 0.1, 1.0)
FLOAT32_MAXIMUM = 3.4028234663852886e38
CROSS_FITTED_GATE_DEPENDENCY_VERSIONS = {
    "keras": "3.15.0",
    "numpy": "2.2.6",
    "tensorflow": "2.20.0",
    "xgboost": "3.1.2",
}
CROSS_FITTED_GATE_SCIKIT_LEARN_VERSION = "1.7.2"

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
TAIL_AWARE_NEURAL_NETWORK_DOMAIN = {
    **NEURAL_NETWORK_DOMAIN,
    "target_weighting": ("none", "sliced_resin_mass_band_1_2_3_4"),
}
TAIL_AWARE_XGBOOST_DOMAIN = {
    **XGBOOST_DOMAIN,
    "target_weighting": ("none", "sliced_resin_mass_band_1_2_3_4"),
}
LARGE_BATCH_NEURAL_NETWORK_DOMAIN = {
    **TAIL_AWARE_NEURAL_NETWORK_DOMAIN,
    "batch_size": (512, 1024),
    "maximum_epochs": (150, 200),
    "early_stopping_patience": (12, 16),
}
LARGE_BATCH_XGBOOST_DOMAIN = {
    **TAIL_AWARE_XGBOOST_DOMAIN,
    "n_estimators": (1500, 1800, 2400),
}
TAIL_ALIGNED_NEURAL_NETWORK_DOMAIN = {
    **NEURAL_NETWORK_DOMAIN,
    "loss": ("huber",),
    "validation_selection": (TAIL_ALIGNED_VALIDATION_SELECTION,),
}
TAIL_ALIGNED_XGBOOST_DOMAIN = {
    **XGBOOST_DOMAIN,
    "validation_selection": (TAIL_ALIGNED_VALIDATION_SELECTION,),
}

CANDIDATE_RUNTIME_CONTRACT: dict[str, dict[str, Any]] = {
    "neural_network": {
        "feature_dtype": "float32",
        "output_layer": {"units": 1, "activation": "linear"},
        "normalization": "fitted_on_fold_training_records_only",
        "early_stopping_monitor": "val_mean_absolute_error",
        "early_stopping_mode": "min",
        "restore_best_weights": True,
        "shuffle_training_records": True,
    },
    "xgboost": {
        "feature_dtype": "float32",
        "tree_method": "hist",
        "eval_metric": "mae",
        "random_state": "evaluation_seed",
        "early_stopping_partition": "fold_validation_only",
    },
}


@dataclass(frozen=True)
class SearchPlanPolicy:
    plan_kind: str
    prediction_features: tuple[str, ...]
    feature_transformation_version: str
    fixed_seeds: tuple[int, int] | None
    neural_network_trials: int
    xgboost_trials: int
    ensemble_trials: int
    second_seed_candidates: int
    maximum_candidate_runs: int
    maximum_elapsed_seconds: float
    neural_network_domain: Mapping[str, tuple[Any, ...]]
    xgboost_domain: Mapping[str, tuple[Any, ...]]
    hypothesis: str


@dataclass(frozen=True)
class SearchLimits:
    seed: int
    second_seed: int | None = None
    plan_kind: str = "baseline"
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

    @classmethod
    def for_plan(
        cls, seed: int, plan_kind: str = "baseline", second_seed: int | None = None,
    ) -> "SearchLimits":
        policy = _plan_policy(plan_kind)
        return cls(
            seed=seed, second_seed=second_seed, plan_kind=policy.plan_kind,
            neural_network_trials=policy.neural_network_trials,
            xgboost_trials=policy.xgboost_trials,
            ensemble_trials=policy.ensemble_trials,
            second_seed_candidates=policy.second_seed_candidates,
            maximum_candidate_runs=policy.maximum_candidate_runs,
            maximum_elapsed_seconds=policy.maximum_elapsed_seconds,
        )


SEARCH_PLAN_POLICIES = {
    "baseline": SearchPlanPolicy(
        "baseline", LEGACY_FEATURES, TRANSFORMATION_VERSION, None,
        6, 6, 3, 5, 20, 7200.0,
        NEURAL_NETWORK_DOMAIN, XGBOOST_DOMAIN,
        "baseline governed candidate search",
    ),
    "tail_aware_expanded": SearchPlanPolicy(
        "tail_aware_expanded", LEGACY_FEATURES, TRANSFORMATION_VERSION, None,
        12, 12, 6, 10, 40, 7200.0,
        TAIL_AWARE_NEURAL_NETWORK_DOMAIN, TAIL_AWARE_XGBOOST_DOMAIN,
        "bounded sliced resin mass weighting and broader configuration coverage "
        "reduce serious absolute errors",
    ),
    "large_batch_extended": SearchPlanPolicy(
        "large_batch_extended", LEGACY_FEATURES, TRANSFORMATION_VERSION, None,
        24, 24, 12, 20, 80, 14_400.0,
        LARGE_BATCH_NEURAL_NETWORK_DOMAIN, LARGE_BATCH_XGBOOST_DOMAIN,
        "larger neural-network batches, longer neural-network training, and "
        "higher XGBoost tree ceilings with denser deterministic coverage reduce "
        "serious absolute errors",
    ),
    "geometry_regime": SearchPlanPolicy(
        "geometry_regime", GEOMETRY_REGIME_FEATURES,
        GEOMETRY_REGIME_TRANSFORMATION_VERSION, (41, 42),
        12, 12, 6, 10, 40, 7200.0,
        LARGE_BATCH_NEURAL_NETWORK_DOMAIN, LARGE_BATCH_XGBOOST_DOMAIN,
        "a richer deterministic source-neutral geometry representation with "
        "logarithmic, ratio, and orientation-invariant features reduces serious "
        "absolute errors",
    ),
    "legacy_geometry_augmentation": SearchPlanPolicy(
        "legacy_geometry_augmentation", LEGACY_GEOMETRY_AUGMENTATION_FEATURES,
        LEGACY_GEOMETRY_AUGMENTATION_TRANSFORMATION_VERSION, (41, 42),
        6, 6, 3, 5, 20, 7200.0,
        NEURAL_NETWORK_DOMAIN, XGBOOST_DOMAIN,
        "augmenting the retained legacy candidate inputs with four deterministic "
        "source-neutral compactness, shape, and curvature terms reduces serious "
        "absolute errors",
    ),
    "cross_fitted_geometry_gate": SearchPlanPolicy(
        "cross_fitted_geometry_gate", LEGACY_FEATURES, TRANSFORMATION_VERSION,
        (41, 42), 6, 6, 3, 5, 20, 7200.0,
        NEURAL_NETWORK_DOMAIN, XGBOOST_DOMAIN,
        "a constrained geometry-conditioned gate fitted from training-only "
        "out-of-fold base-model predictions exploits complementary errors enough "
        "to satisfy the unchanged serious-error gates",
    ),
    "tail_aligned_selection": SearchPlanPolicy(
        "tail_aligned_selection", LEGACY_FEATURES, TRANSFORMATION_VERSION,
        (41, 42), 6, 6, 3, 5, 20, 7200.0,
        TAIL_ALIGNED_NEURAL_NETWORK_DOMAIN, TAIL_ALIGNED_XGBOOST_DOMAIN,
        "selecting bounded model checkpoints, component pairs, and ensemble "
        "weights by fixed serious-error gate violation before the unchanged "
        "ranking metrics reduces serious absolute errors",
    ),
}


def _plan_policy(plan_kind: str) -> SearchPlanPolicy:
    try:
        return SEARCH_PLAN_POLICIES[plan_kind]
    except (KeyError, TypeError):
        raise ValueError("invalid_search_plan") from None


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    parameters: dict[str, Any]
    ordered_prediction_features: tuple[str, ...] = LEGACY_FEATURES
    feature_transformation_version: str = TRANSFORMATION_VERSION

    @property
    def model_kind(self) -> ModelKind | None:
        """Architecture kind; `family` is retained for evidence compatibility."""
        try:
            return ModelKind(self.family)
        except ValueError:
            return None

    @property
    def specification(self):
        if self.family not in {"neural_network", "xgboost"}:
            return None
        return candidate_model_specification(
            self.family, self.parameters,
            ordered_prediction_features=self.ordered_prediction_features,
            identity_namespace=(
                f"minires-model-definition-v1:{self.feature_transformation_version}"
                if self.ordered_prediction_features != LEGACY_FEATURES
                else "minires-model-definition-v1"
            ),
        )


@dataclass(frozen=True)
class DeclaredCandidate:
    """One complete model configuration with a content-derived identity."""

    family: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        copied = _freeze_json_lists(copy.deepcopy(dict(self.parameters)))
        if not isinstance(copied, dict):
            raise TypeError("candidate parameters must be a mapping")
        object.__setattr__(self, "parameters", MappingProxyType(copied))

    @property
    def model_kind(self) -> ModelKind:
        """Architecture kind, not the miniature-family grouping concept."""
        try:
            return ModelKind(self.family)
        except ValueError:
            raise ValueError("invalid_candidate_configuration") from None

    @property
    def specification(self):
        return candidate_model_specification(self.family, self.parameters)

    @property
    def candidate_id(self) -> str:
        payload = {
            "version": TUNING_VERSION,
            "family": self.family,
            "parameters": dict(self.parameters),
            "features": list(LEGACY_FEATURES),
            "transformation_version": TRANSFORMATION_VERSION,
            "runtime_configuration": CANDIDATE_RUNTIME_CONTRACT.get(self.family),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return f"declared-{self.family}-{sha256(encoded).hexdigest()[:16]}"

    def to_dict(self) -> dict[str, Any]:
        preprocessing = (
            "normalization_fitted_on_fold_training_records_only"
            if self.family == "neural_network" else "unnormalized_float32"
        )
        specification = self.specification
        return {
            "version": TUNING_VERSION,
            "candidate_id": self.candidate_id,
            "family": self.family,
            "model_kind": self.model_kind.value,
            "model_specification": specification.to_dict(),
            "parameters": json.loads(json.dumps(dict(self.parameters), allow_nan=False)),
            "features": list(LEGACY_FEATURES),
            "transformation_version": TRANSFORMATION_VERSION,
            "preprocessing": preprocessing,
            "runtime_configuration": copy.deepcopy(
                CANDIDATE_RUNTIME_CONTRACT.get(self.family)
            ),
            "output_unit": "g",
        }

    def as_runtime_candidate(self) -> Candidate:
        parameters = _freeze_json_lists(self.to_dict()["parameters"])
        assert isinstance(parameters, dict)
        return Candidate(self.candidate_id, self.family, parameters)


@dataclass(frozen=True)
class SearchPlanIdentities:
    normalized_input_fingerprint: str
    source_allocation_fingerprint: str
    code_fingerprint: str
    configuration_fingerprint: str
    code_configuration_fingerprint: str


@dataclass(frozen=True)
class SearchPlan:
    version: str
    generator_version: str
    plan_id: str
    prediction_features: tuple[str, ...]
    feature_transformation_version: str
    seed: int
    second_seed: int
    input_fingerprint: str
    normalized_input_fingerprint: str
    source_allocation_fingerprint: str
    code_fingerprint: str
    configuration_fingerprint: str
    code_configuration_fingerprint: str
    dependency_versions: dict[str, str]
    dependency_contract: dict[str, Any]
    parameter_domains: dict[str, dict[str, tuple[Any, ...]]]
    generator: dict[str, Any]
    component_trials: tuple[Candidate, ...]
    ensemble_rules: tuple[dict[str, Any], ...]
    eligibility_gates: dict[str, float | int]
    ranking_rule: tuple[str, ...]
    second_seed_rule: dict[str, Any]
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
        *, prediction_ranker: Callable[[Sequence[float]], tuple[Any, ...]] | None = None,
    ) -> CandidateFoldFit: ...

    def refit(
        self, candidate: Candidate, seed: int,
        features: Sequence[tuple[float, ...]], targets: Sequence[float],
        fixed_training_counts: Mapping[str, int | float],
    ) -> LockedFit: ...

    def load_locked(
        self, candidate: Candidate, directory: Path, contract: Mapping[str, Any]
    ) -> Callable[[Sequence[tuple[float, ...]]], Sequence[float]]: ...

    def serialize_fold(self, fitted: CandidateFoldFit) -> Mapping[str, bytes]: ...


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
    artifact_checksums: dict[str, str] | None = None


@dataclass(frozen=True)
class DeclaredCandidateEvaluation:
    status: str
    blockers: tuple[str, ...]
    contract: dict[str, Any]
    seed: int
    metrics: dict[str, float | int | None]
    source_reports: tuple[dict[str, Any], ...]
    fit_metadata: tuple[dict[str, Any], ...]
    dependency_versions: dict[str, str]
    resource_use: dict[str, Any]
    artifact_checksums: dict[str, str]

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "blockers": list(self.blockers),
            "candidate_id": self.contract.get("candidate_id"),
            "family": self.contract.get("family"),
            "metrics": dict(self.metrics),
            "resource_use": dict(self.resource_use),
            "classification": "private_candidate_evaluation",
        }
        if public:
            return result
        result.update({
            "candidate_contract": self.contract,
            "seed": self.seed,
            "dependency_versions": dict(self.dependency_versions),
            "source_reports": list(self.source_reports),
            "partition_audits": [
                {
                    "source": report["source"],
                    "train_rows": report["train_rows"],
                    "validation_rows": report["validation_rows"],
                    "test_rows": report["test_rows"],
                    "test_data_fingerprint": report["test_data_fingerprint"],
                }
                for report in self.source_reports
            ],
            "fit_metadata": list(self.fit_metadata),
            "artifact_checksums": dict(self.artifact_checksums),
        })
        return result


@dataclass(frozen=True)
class GeometryGateModelSpecification:
    neural_network: ModelSpecification
    xgboost: ModelSpecification
    ordered_prediction_features: tuple[str, ...]
    gate_features: tuple[str, ...]
    cross_fit_folds: int
    ridge_penalty: float
    identity_namespace: str = CROSS_FITTED_GEOMETRY_GATE_TRANSFORMATION_VERSION

    @property
    def stable_identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_identity=False), sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode()
        return f"model-geometry_gate-{sha256(encoded).hexdigest()[:16]}"

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        result = {
            "version": self.identity_namespace,
            "model_kind": "geometry_gate",
            "ordered_prediction_features": list(self.ordered_prediction_features),
            "preprocessing": {
                "feature_dtype": "float32",
                "normalization": "defined_by_members_and_training_oof_gate",
            },
            "architecture_parameters": {
                "gate_features": list(self.gate_features),
                "cross_fit_folds": self.cross_fit_folds,
                "ridge_penalty": self.ridge_penalty,
                "weight_constraint": "clip_0_1",
            },
            "training_parameters": {
                "gate_fit_partition": "training_oof_predictions_only",
            },
            "members": {
                "neural_network": self.neural_network.to_dict(),
                "xgboost": self.xgboost.to_dict(),
            },
            "output_unit": "g",
        }
        if include_identity:
            result["stable_identity"] = self.stable_identity
        return result


@dataclass(frozen=True)
class LockedCandidate:
    candidate: Candidate
    predictor: Callable[[Sequence[tuple[float, ...]]], Sequence[float]]
    directory: Path
    manifest: dict[str, Any]
    contract: dict[str, Any]

    @property
    def specification(self) -> ModelSpecification | GeometryGateModelSpecification:
        return _locked_model_specification(
            self.candidate, self.contract["fixed_training_counts"]
        )

    def load_predictor(self, runtime: CandidateRuntime):
        """Load the locked predictor without exposing its architecture to assessment."""
        return runtime.load_locked(self.candidate, self.directory, self.contract)


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
        skipped = _skipped_candidates(self)
        base.update({
            "search_plan": self.plan.to_dict(),
            "control_result": _run_to_dict(self.control_result),
            "initial_results": [_run_to_dict(item) for item in self.initial_results],
            "second_seed_results": [_run_to_dict(item) for item in self.second_seed_results],
            "second_seed_comparison": _second_seed_comparison(self),
            "combined_results": list(self.combined_results),
            "initial_promotable_ranking": _initial_promotable_ranking(self),
            "promotable_ranking": _promotable_ranking(self),
            "best_development_result": _best_development_result(self),
            "selected_ensemble_weights": _selected_ensemble_weights(self.initial_results),
            "skipped_candidates": skipped,
            "candidate_history": _candidate_history(self, skipped),
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
        if manifest.get("version") != TUNING_VERSION or manifest.get("create_only") is not True:
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
        expected_dependencies = dict(sorted(dependency_versions.items()))
        dependency_environment = contract.get("dependency_environment")
        if (
            contract.get("dependency_versions") != expected_dependencies
            or not isinstance(dependency_environment, Mapping)
            or dependency_environment.get("versions") != expected_dependencies
        ):
            blockers.add("locked_candidate_dependency_mismatch")
        preprocessing = json.loads((root / "preprocessing-state.json").read_text())
        if not isinstance(preprocessing, dict) or not _valid_locked_contract(contract):
            blockers.add("locked_candidate_contract_mismatch")
        if contract.get("candidate", {}).get("family") == "geometry_gate":
            gate_state = json.loads((root / "gate-state.json").read_text())
            if (
                not isinstance(gate_state, dict)
                or not _valid_geometry_gate_state(gate_state)
                or preprocessing.get("gate") != gate_state
            ):
                blockers.add("locked_candidate_contract_mismatch")
    except (AttributeError, OSError, ValueError, TypeError, json.JSONDecodeError):
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


def create_search_plan(
    records: Dataset,
    config: EvaluationConfig,
    *,
    limits: SearchLimits,
    dependency_versions: Mapping[str, str],
    output_root: str | Path | None = None,
    parameter_domains: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
) -> SearchPlan:
    """Build and optionally persist a private plan without fitting or scoring."""
    loaded, input_fingerprint = load_records(records)
    rows = normalize(loaded, config, contract="legacy")
    identities = _search_plan_identities(rows, config)
    plan = _generate_bound_search_plan(
        limits, input_fingerprint, dependency_versions, identities,
        parameter_domains=parameter_domains,
    )
    if output_root is None:
        return plan
    output = Path(output_root)
    if "private" not in output.resolve().parts:
        raise ValueError("search_plan_output_unavailable")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        write_private_json(output / "search-plan.json", plan.to_dict())
        plan_path = output / "search-plan.json"
        write_private_json(output / "manifest.json", {
            "version": TUNING_VERSION,
            "artifacts": {"search-plan.json": sha256(plan_path.read_bytes()).hexdigest()},
            "create_only": True,
            "publication_performed": False,
        })
    except (InputError, OSError, TypeError, ValueError):
        raise ValueError("search_plan_output_unavailable") from None
    return plan


def _generate_bound_search_plan(
    limits: SearchLimits,
    input_fingerprint: str,
    dependency_versions: Mapping[str, str],
    identities: SearchPlanIdentities,
    *,
    parameter_domains: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
) -> SearchPlan:
    return generate_search_plan(
        limits,
        input_fingerprint=input_fingerprint,
        normalized_input_fingerprint=identities.normalized_input_fingerprint,
        source_allocation_fingerprint=identities.source_allocation_fingerprint,
        code_fingerprint=identities.code_fingerprint,
        configuration_fingerprint=identities.configuration_fingerprint,
        code_configuration_fingerprint=identities.code_configuration_fingerprint,
        dependency_versions=dependency_versions,
        parameter_domains=parameter_domains,
    )


def generate_search_plan(
    limits: SearchLimits,
    *,
    input_fingerprint: str,
    code_fingerprint: str,
    dependency_versions: Mapping[str, str],
    normalized_input_fingerprint: str | None = None,
    source_allocation_fingerprint: str | None = None,
    configuration_fingerprint: str | None = None,
    code_configuration_fingerprint: str | None = None,
    parameter_domains: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
) -> SearchPlan:
    """Generate the complete finite component plan without inspecting labels or scores."""
    _validate_limits(limits)
    if (
        not dependency_versions
        or any(
            not isinstance(name, str) or not name
            or not isinstance(version, str) or not version
            for name, version in dependency_versions.items()
        )
        or any(
            not isinstance(value, str) or not value
            for value in (
                input_fingerprint, code_fingerprint,
                normalized_input_fingerprint or input_fingerprint,
                source_allocation_fingerprint or "derived",
                configuration_fingerprint or code_fingerprint,
                code_configuration_fingerprint or "derived",
            )
        )
    ):
        raise ValueError("invalid_search_plan")
    policy = _plan_policy(limits.plan_kind)
    domains = _validated_parameter_domains(parameter_domains, policy)
    neural = _space_filling_candidates(
        "neural_network", domains["neural_network"], limits.neural_network_trials,
        limits.seed, policy.prediction_features, policy.feature_transformation_version,
    )
    xgboost = _space_filling_candidates(
        "xgboost", domains["xgboost"], limits.xgboost_trials, limits.seed ^ 0x5EED,
        policy.prediction_features, policy.feature_transformation_version,
    )
    components = neural + xgboost
    if len({candidate.candidate_id for candidate in components}) != len(components):
        raise ValueError("invalid_search_plan")
    for candidate in components:
        _validate_declared_candidate(candidate)
    normalized_identity = normalized_input_fingerprint or input_fingerprint
    allocation_identity = source_allocation_fingerprint or fingerprint({
        "allocation_version": ALLOCATION_VERSION,
        "input_fingerprint": normalized_identity,
    })
    configuration_identity = configuration_fingerprint or code_fingerprint
    combined_identity = code_configuration_fingerprint or fingerprint({
        "code": code_fingerprint, "configuration": configuration_identity,
    })
    dependencies = dict(sorted(dependency_versions.items()))
    if policy.plan_kind == "cross_fitted_geometry_gate":
        ensemble_rules = tuple({
            "ensemble_slot": rank,
            "component_rank": rank,
            "construction": "cross_fitted_geometry_gate",
            "pairing": "rank_each_component_family_then_pair_equal_rank",
            "component_ranking_partition": "development_validation_only",
            "cross_fit_folds": CROSS_FIT_FOLDS,
            "cross_fit_partition": "training_records_only",
            "cross_fit_assignment": "stable_record_identity_hash_without_source_metadata",
            "gate_fit_partition": "training_oof_predictions_only",
            "validation_use": "scoring_eligibility_ranking_and_locking_only",
            "gate_features": GEOMETRY_REGIME_FEATURES,
            "gate_feature_transformation_version": GEOMETRY_REGIME_TRANSFORMATION_VERSION,
            "gate_transformation_version": (
                CROSS_FITTED_GEOMETRY_GATE_TRANSFORMATION_VERSION
            ),
            "ridge_penalty": GEOMETRY_GATE_RIDGE_PENALTIES[rank - 1],
            "weight_constraint": "clip_0_1",
            "stable_tie_breaker": "candidate_id_ascending",
        } for rank in range(1, limits.ensemble_trials + 1))
    else:
        ensemble_rules = tuple({
            "ensemble_slot": rank,
            "component_rank": rank,
            "pairing": "rank_each_component_family_then_pair_equal_rank",
            "component_ranking_partition": "development_validation_only",
            "weight_selection_partition": "development_validation_only",
            "neural_network_weight_grid": limits.ensemble_neural_network_weights,
            "selection_rule": (
                TAIL_ALIGNED_VALIDATION_SELECTION
                if policy.plan_kind == "tail_aligned_selection" else "pooled_mae"
            ),
            "stable_tie_breaker": "candidate_id_ascending",
        } for rank in range(1, limits.ensemble_trials + 1))
    plan_payload: dict[str, Any] = {
        "version": TUNING_VERSION,
        "generator_version": TUNING_VERSION,
        "prediction_features": policy.prediction_features,
        "feature_transformation_version": policy.feature_transformation_version,
        "seed": limits.seed,
        "second_seed": limits.resolved_second_seed,
        "input_fingerprint": input_fingerprint,
        "normalized_input_fingerprint": normalized_identity,
        "source_allocation_fingerprint": allocation_identity,
        "code_fingerprint": code_fingerprint,
        "configuration_fingerprint": configuration_identity,
        "code_configuration_fingerprint": combined_identity,
        "dependency_versions": dependencies,
        "dependency_contract": {
            "python_supported": ">=3.11,<3.14",
            "versions_must_match": True,
            "versions": dependencies,
        },
        "parameter_domains": domains,
        "generator": {
            "strategy": "seeded_mixed_radix_space_filling",
            "ordered": True,
            "target_access": False,
            "prior_score_access": False,
            "plan_kind": policy.plan_kind,
            "hypothesis": policy.hypothesis,
            "validation_selection": (
                TAIL_ALIGNED_VALIDATION_SELECTION
                if policy.plan_kind == "tail_aligned_selection" else "pooled_mae"
            ),
        },
        "component_trials": components,
        "ensemble_rules": ensemble_rules,
        "eligibility_gates": _eligibility_gates(),
        "ranking_rule": (
            "eligible_serious_error_gates_first",
            "source_balanced_mae_g_ascending",
            "pooled_mae_g_ascending",
            "pooled_within_2g_fraction_descending",
            "stable_candidate_id_ascending",
        ),
        "second_seed_rule": {
            "candidate_count": limits.second_seed_candidates,
            "selection": "best_eligible_initial_candidates_by_ranking_rule",
            "combination": "equal_seed_weight",
            "unfavorable_repetitions_retained": True,
            "both_seed_results_must_be_eligible": True,
        },
        "resource_limits": {
            "maximum_candidate_runs": limits.maximum_candidate_runs,
            "maximum_elapsed_seconds": limits.maximum_elapsed_seconds,
            "initial_neural_network_trials": limits.neural_network_trials,
            "initial_xgboost_trials": limits.xgboost_trials,
            "initial_ensemble_trials": limits.ensemble_trials,
            "second_seed_candidates": limits.second_seed_candidates,
        },
        "control": {
            "classification": "clean_fixed_configuration_baseline",
            "candidate_id": "clean-fixed-control",
            "configuration": asdict(LearnedBaselineConfig()),
            "model_specification": fixed_model_specification().to_dict(),
            "candidate_slot_consumed": False,
            "configuration_mutable": False,
        },
    }
    plan_id = "search-plan-" + fingerprint(_jsonable(plan_payload))[:24]
    return SearchPlan(plan_id=plan_id, **plan_payload)


def _validated_parameter_domains(
    supplied: Mapping[str, Mapping[str, Sequence[Any]]] | None,
    policy: SearchPlanPolicy,
) -> dict[str, dict[str, tuple[Any, ...]]]:
    supported = _supported_domains(policy)
    source = supported if supplied is None else supplied
    if set(source) != set(supported):
        raise ValueError("invalid_search_plan")
    domains: dict[str, dict[str, tuple[Any, ...]]] = {}
    for family, expected_domain in supported.items():
        proposed = source[family]
        if set(proposed) != set(expected_domain):
            raise ValueError("invalid_search_plan")
        domains[family] = {}
        for name in sorted(proposed):
            values = proposed[name]
            if isinstance(values, (str, bytes)):
                raise ValueError("invalid_search_plan")
            choices = tuple(_freeze_json_lists(value) for value in values)
            if not choices or any(
                not _supported_domain_value(value, expected_domain[name]) for value in choices
            ):
                raise ValueError("invalid_search_plan")
            domains[family][name] = choices
    return domains


def _supported_domains(
    policy: SearchPlanPolicy,
) -> dict[str, dict[str, tuple[Any, ...]]]:
    return {
        "neural_network": dict(policy.neural_network_domain),
        "xgboost": dict(policy.xgboost_domain),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Candidate):
        return asdict(value)
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _search_plan_identities(
    rows: Sequence[CanonicalRow], config: EvaluationConfig,
) -> SearchPlanIdentities:
    code_identity = code_fingerprint()
    configuration_identity = fingerprint(asdict(config))
    normalized_identity = fingerprint([asdict(row) for row in rows])
    allocation_identity = fingerprint({
        "allocation_version": ALLOCATION_VERSION,
        "transformation_version": TRANSFORMATION_VERSION,
        "configuration": asdict(config),
        "grouping_evidence": [
            {
                "row_index": row.row_index,
                "outcome": row.outcome,
                "anonymous_source_group": row.metadata.get("anonymous_source_group"),
                "miniature_family": row.metadata.get("miniature_family"),
                "duplicate_group": row.metadata.get("duplicate_group"),
                "geometry_fingerprint": row.metadata.get("geometry_fingerprint"),
                "record_identity": row.metadata.get("record_identity"),
                "location_evidence": row.metadata.get("location_evidence"),
            }
            for row in rows
        ],
    })
    return SearchPlanIdentities(
        normalized_input_fingerprint=normalized_identity,
        source_allocation_fingerprint=allocation_identity,
        code_fingerprint=code_identity,
        configuration_fingerprint=configuration_identity,
        code_configuration_fingerprint=fingerprint({
            "code": code_identity, "configuration": configuration_identity,
        }),
    )


def _space_filling_candidates(
    family: str, domain: Mapping[str, tuple[Any, ...]], count: int, seed: int,
    ordered_prediction_features: tuple[str, ...] = LEGACY_FEATURES,
    feature_transformation_version: str = TRANSFORMATION_VERSION,
) -> tuple[Candidate, ...]:
    rng = random.Random(seed)
    keys = tuple(sorted(domain))
    combination_count = math.prod(len(domain[key]) for key in keys)
    if count > combination_count:
        raise ValueError("invalid_search_plan")
    offset = rng.randrange(combination_count)
    step = rng.randrange(1, combination_count + 1)
    while math.gcd(step, combination_count) != 1:
        step = 1 if step == combination_count else step + 1
    candidates = []
    for trial in range(count):
        encoded = (offset + trial * step) % combination_count
        parameters = {}
        for key in keys:
            choices = domain[key]
            parameters[key] = choices[encoded % len(choices)]
            encoded //= len(choices)
        identity_payload = [family, parameters]
        if ordered_prediction_features != LEGACY_FEATURES:
            identity_payload.extend(
                (ordered_prediction_features, feature_transformation_version)
            )
        identity = json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":"),
        )
        candidates.append(Candidate(
            candidate_id=f"{family[:3]}-{trial + 1:02d}-{sha256(identity.encode()).hexdigest()[:8]}",
            family=family,
            parameters=parameters,
            ordered_prediction_features=ordered_prediction_features,
            feature_transformation_version=feature_transformation_version,
        ))
    return tuple(candidates)


def evaluate_declared_candidate(
    records: Dataset,
    config: EvaluationConfig,
    *,
    candidate: DeclaredCandidate,
    runtime: CandidateRuntime | None = None,
    seed: int,
    output_root: str | Path | None = None,
) -> DeclaredCandidateEvaluation:
    """Evaluate one declared NN or XGBoost candidate through source holdouts."""
    started = time.perf_counter()
    cpu_started = time.process_time()
    if runtime is None:
        try:
            runtime = TensorflowXGBoostCandidateRuntime()
        except ImportError:
            runtime = _BlockedCandidateRuntime("candidate_evaluation_dependencies_required")
        except RuntimeError:
            runtime = _BlockedCandidateRuntime("candidate_evaluation_runtime_unavailable")
    dependencies = dict(sorted(getattr(runtime, "dependency_versions", {}).items()))
    try:
        contract = candidate.to_dict()
        runtime_candidate = candidate.as_runtime_candidate()
        _validate_declared_candidate(runtime_candidate)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("invalid candidate seed")
    except (AssertionError, TypeError, ValueError, OverflowError):
        return DeclaredCandidateEvaluation(
            "blocked", ("invalid_candidate_configuration",),
            {"version": TUNING_VERSION, "candidate_id": None,
             "family": candidate.family if isinstance(candidate.family, str) else None},
            seed, _empty_metrics(), (), (), dependencies,
            _resources(time.perf_counter() - started, time.process_time() - cpu_started), {},
        )

    startup_blockers = tuple(sorted(set(getattr(runtime, "startup_blockers", ()))))
    if startup_blockers:
        return DeclaredCandidateEvaluation(
            "blocked", startup_blockers, contract, seed, _empty_metrics(), (), (),
            dependencies,
            _resources(time.perf_counter() - started, time.process_time() - cpu_started), {},
        )

    if output_root is None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "private" / "declared-candidate"
            root.mkdir(parents=True, mode=0o700)
            return _evaluate_declared_at_root(
                records, config, runtime_candidate, runtime, seed, contract, root,
                dependencies, started, cpu_started, persist=False,
            )

    root = Path(output_root)
    if "private" not in root.resolve().parts:
        return DeclaredCandidateEvaluation(
            "blocked", ("candidate_artifact_failed",), contract, seed, _empty_metrics(),
            (), (), dependencies,
            _resources(time.perf_counter() - started, time.process_time() - cpu_started), {},
        )
    try:
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        return DeclaredCandidateEvaluation(
            "blocked", ("candidate_artifact_failed",), contract, seed, _empty_metrics(),
            (), (), dependencies,
            _resources(time.perf_counter() - started, time.process_time() - cpu_started), {},
        )
    return _evaluate_declared_at_root(
        records, config, runtime_candidate, runtime, seed, contract, root,
        dependencies, started, cpu_started, persist=True,
    )


def _evaluate_declared_at_root(
    records: Dataset, config: EvaluationConfig, candidate: Candidate,
    runtime: CandidateRuntime, seed: int, contract: dict[str, Any], root: Path,
    dependencies: dict[str, str], started: float, cpu_started: float, *, persist: bool,
) -> DeclaredCandidateEvaluation:
    try:
        loaded, input_fingerprint = load_records(records)
        rows = normalize(loaded, config, contract="legacy")
        manifest = freeze_splits(
            rows, input_fingerprint, asdict(config), root / "partition-audit.json"
        )
    except (InputError, OSError, TypeError, ValueError):
        result = DeclaredCandidateEvaluation(
            "blocked", ("invalid_candidate_partitions",), contract, seed,
            _empty_metrics(), (), (), dependencies,
            _resources(time.perf_counter() - started, time.process_time() - cpu_started), {},
        )
        return _persist_declared_result(root, result) if persist else result
    if manifest["status"] != "frozen_source_holdout":
        result = DeclaredCandidateEvaluation(
            "blocked", tuple(manifest["blockers"]), contract, seed, _empty_metrics(),
            (), (), dependencies,
            _resources(time.perf_counter() - started, time.process_time() - cpu_started), {},
        )
        return _persist_declared_result(root, result) if persist else result

    by_index = {row.row_index: row for row in rows if row.outcome == "included"}
    run = _evaluate_candidate(
        candidate, seed, runtime, manifest, by_index, (),
        artifact_root=(root / "fold-artifacts") if persist else None,
    )
    result = DeclaredCandidateEvaluation(
        run.status, run.blockers, contract, seed, run.metrics, run.source_reports,
        run.fit_metadata, dependencies,
        _resources(time.perf_counter() - started, time.process_time() - cpu_started),
        dict(run.artifact_checksums or {}),
    )
    return _persist_declared_result(root, result) if persist else result


def _persist_declared_result(
    root: Path, result: DeclaredCandidateEvaluation
) -> DeclaredCandidateEvaluation:
    try:
        write_private_json(root / "candidate-report.json", result.to_dict())
        inventory = {
            str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        write_private_json(root / "manifest.json", {
            "version": TUNING_VERSION,
            "artifacts": inventory,
            "create_only": True,
            "publication_performed": False,
        })
        return result
    except (InputError, OSError, TypeError, ValueError):
        return DeclaredCandidateEvaluation(
            "blocked", ("candidate_artifact_failed",), result.contract, result.seed,
            result.metrics, result.source_reports, result.fit_metadata,
            result.dependency_versions, result.resource_use, result.artifact_checksums,
        )


def develop_candidates(
    training_records: Dataset,
    validation_records: Dataset,
    config: EvaluationConfig,
    *,
    runtime: CandidateRuntime,
    output_root: str | Path,
    limits: SearchLimits,
    clock: Callable[[], float] = time.monotonic,
) -> TuningResult:
    """Select and lock a candidate from explicit train and validation artifacts."""
    output = Path(output_root)
    if "private" not in output.resolve().parts:
        raise InputError("private_output_directory_required")
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError:
        raise InputError("private_output_directory_unavailable") from None

    training_loaded, training_fingerprint = load_records(training_records)
    validation_loaded, validation_fingerprint = load_records(validation_records)
    training_rows = normalize(training_loaded, config, contract="legacy")
    validation_rows = normalize(validation_loaded, config, contract="legacy")
    combined_fingerprint = fingerprint({
        "training": training_fingerprint,
        "validation": validation_fingerprint,
    })
    identities = SearchPlanIdentities(
        normalized_input_fingerprint=fingerprint({
            "training": [asdict(row) for row in training_rows],
            "validation": [asdict(row) for row in validation_rows],
        }),
        source_allocation_fingerprint=fingerprint({
            "contract": "explicit_train_validation",
            "training": training_fingerprint,
            "validation": validation_fingerprint,
        }),
        code_fingerprint=code_fingerprint(),
        configuration_fingerprint=fingerprint(asdict(config)),
        code_configuration_fingerprint="",
    )
    identities = _complete_explicit_identities(identities)
    plan = _generate_bound_search_plan(
        limits, combined_fingerprint, runtime.dependency_versions, identities,
    )
    write_private_json(output / "search-plan.json", plan.to_dict())

    if not _valid_explicit_partitions(training_rows, validation_rows):
        result = TuningResult(
            "blocked", ("invalid_or_overlapping_partition_identities",), 0,
            {"neural_network": 0, "xgboost": 0, "ensemble": 0,
             "second_seed": 0, "control": 0},
            plan, None, (), (), (), None, _resources(0.0, 0.0),
        )
        _write_tuning_outputs(output, result)
        return result

    required_prediction_features = (
        CROSS_FITTED_GEOMETRY_GATE_FEATURES
        if limits.plan_kind == "cross_fitted_geometry_gate"
        else plan.prediction_features
    )
    if not _valid_candidate_feature_data(
        (*training_rows, *validation_rows), required_prediction_features
    ):
        result = TuningResult(
            "blocked", ("invalid_candidate_feature_data",), 0,
            {"neural_network": 0, "xgboost": 0, "ensemble": 0,
             "second_seed": 0, "control": 0},
            plan, None, (), (), (), None, _resources(0.0, 0.0),
        )
        _write_tuning_outputs(output, result)
        return result

    startup_blockers = tuple(sorted(set(getattr(runtime, "startup_blockers", ()))))
    if startup_blockers:
        result = TuningResult(
            "blocked", startup_blockers, 0,
            {"neural_network": 0, "xgboost": 0, "ensemble": 0,
             "second_seed": 0, "control": 0},
            plan, None, (), (), (), None, _resources(0.0, 0.0),
        )
        _write_tuning_outputs(output, result)
        return result

    started = clock()
    cpu_started = time.process_time()
    deadline = started + limits.maximum_elapsed_seconds
    allocation = {"neural_network": 0, "xgboost": 0, "ensemble": 0,
                  "second_seed": 0, "control": 0}
    blockers: list[str] = []
    run_count = 0
    control = Candidate("clean-fixed-control", "control", asdict(LearnedBaselineConfig()))
    control_result = _evaluate_explicit_candidate(
        control, LearnedBaselineConfig().seed, runtime, training_rows, validation_rows,
        limits.ensemble_neural_network_weights,
    )
    allocation["control"] = 1
    if control_result.status != "completed":
        blockers.append("control_evaluation_failed")

    initial: list[CandidateRun] = []
    for candidate in plan.component_trials if not blockers else ():
        now = clock()
        if run_count >= limits.maximum_candidate_runs or now >= deadline:
            blockers.append("candidate_search_deadline_reached" if now >= deadline
                            else "candidate_run_limit_reached")
            break
        run = _evaluate_explicit_candidate(
            candidate, plan.seed, runtime, training_rows, validation_rows,
            limits.ensemble_neural_network_weights,
        )
        initial.append(run)
        allocation[candidate.family] += 1
        run_count += 1
        if run.status != "completed":
            blockers.append("candidate_runtime_failed")
            break

    if len(initial) == len(plan.component_trials) and not blockers:
        component_rank_key = (
            _tail_component_rank_key
            if limits.plan_kind == "tail_aligned_selection" else _rank_key
        )
        neural = sorted(
            (run for run in initial if run.candidate.family == "neural_network"),
            key=component_rank_key,
        )
        xgboost = sorted(
            (run for run in initial if run.candidate.family == "xgboost"),
            key=component_rank_key,
        )
        for index in range(limits.ensemble_trials):
            now = clock()
            if run_count >= limits.maximum_candidate_runs or now >= deadline:
                blockers.append("candidate_search_deadline_reached" if now >= deadline
                                else "candidate_run_limit_reached")
                break
            if limits.plan_kind == "cross_fitted_geometry_gate":
                candidate = _explicit_geometry_gate_candidate(
                    index, neural[index].candidate, xgboost[index].candidate,
                    plan.ensemble_rules[index],
                )
            else:
                candidate = _explicit_ensemble_candidate(
                    index, neural[index].candidate, xgboost[index].candidate,
                    limits.ensemble_neural_network_weights,
                )
            run = _evaluate_explicit_candidate(
                candidate, plan.seed, runtime, training_rows, validation_rows,
                limits.ensemble_neural_network_weights,
            )
            initial.append(run)
            allocation["ensemble"] += 1
            run_count += 1
            if run.status != "completed":
                blockers.append("candidate_runtime_failed")
                break

    second: list[CandidateRun] = []
    ranked_initial = sorted((run for run in initial if run.eligible), key=_rank_key)
    repetition_target = min(limits.second_seed_candidates, len(ranked_initial))
    initial_complete = len(initial) == len(plan.component_trials) + limits.ensemble_trials
    if initial_complete and not blockers:
        for first in ranked_initial[:repetition_target]:
            now = clock()
            if run_count >= limits.maximum_candidate_runs or now >= deadline:
                blockers.append("candidate_search_deadline_reached" if now >= deadline
                                else "candidate_run_limit_reached")
                break
            run = _evaluate_explicit_candidate(
                first.candidate, plan.second_seed, runtime, training_rows, validation_rows,
                limits.ensemble_neural_network_weights,
            )
            second.append(run)
            allocation["second_seed"] += 1
            run_count += 1
            if run.status != "completed":
                blockers.append("candidate_runtime_failed")
                break

    combined = _combine_seed_results(initial, second)
    locked: LockedCandidate | None = None
    complete = (
        initial_complete and len(second) == repetition_target
        and all(run.status == "completed" for run in second) and not blockers
    )
    if complete:
        eligible = [item for item in combined if item["eligible"]]
        if not eligible:
            blockers.append("no_eligible_candidate")
        else:
            selected = min(eligible, key=_combined_rank_key)
            candidate = next(
                run.candidate for run in initial
                if run.candidate.candidate_id == selected["candidate_id"]
            )
            try:
                locked = _refit_and_lock_explicit(
                    candidate, selected, plan, runtime, training_rows, validation_rows,
                    initial, second, training_fingerprint, validation_fingerprint,
                    output / "locked-candidate",
                )
            except Exception:
                blockers.append("candidate_refit_or_lock_failed")
    elapsed = max(0.0, clock() - started)
    status = (
        "completed" if locked is not None else "completed_no_candidate"
        if complete and "no_eligible_candidate" in blockers else "blocked"
    )
    result = TuningResult(
        status, tuple(sorted(set(blockers))), run_count, allocation, plan,
        control_result, tuple(initial), tuple(second), tuple(combined), locked,
        _resources(elapsed, time.process_time() - cpu_started),
    )
    _write_tuning_outputs(output, result)
    return result


def _complete_explicit_identities(identities: SearchPlanIdentities) -> SearchPlanIdentities:
    """Complete the combined code/configuration identity for an explicit plan."""
    return SearchPlanIdentities(
        identities.normalized_input_fingerprint,
        identities.source_allocation_fingerprint,
        identities.code_fingerprint,
        identities.configuration_fingerprint,
        fingerprint({
            "code": identities.code_fingerprint,
            "configuration": identities.configuration_fingerprint,
        }),
    )


def _valid_explicit_partitions(
    training: Sequence[CanonicalRow], validation: Sequence[CanonicalRow],
) -> bool:
    if not training or not validation or any(
        row.outcome != "included"
        or not row.metadata.get("record_identity")
        or not row.metadata.get("anonymous_source_group")
        for row in (*training, *validation)
    ):
        return False
    training_ids = [str(row.metadata["record_identity"]) for row in training]
    validation_ids = [str(row.metadata["record_identity"]) for row in validation]
    linkage_fields = ("duplicate_group", "geometry_fingerprint", "location_evidence")
    cross_partition_linkage = any(
        {str(row.metadata[field]) for row in training if row.metadata.get(field)}
        & {str(row.metadata[field]) for row in validation if row.metadata.get(field)}
        for field in linkage_fields
    )
    return (
        len(training_ids) == len(set(training_ids))
        and len(validation_ids) == len(set(validation_ids))
        and not set(training_ids) & set(validation_ids)
        and not cross_partition_linkage
    )


def _explicit_geometry_gate_candidate(
    index: int, neural: Candidate, xgboost: Candidate, rule: Mapping[str, Any],
) -> Candidate:
    if (
        neural.ordered_prediction_features != LEGACY_FEATURES
        or xgboost.ordered_prediction_features != LEGACY_FEATURES
    ):
        raise ValueError("invalid_candidate_feature_contract")
    parameters = {
        "component_rank": index + 1,
        "selection_partition": "validation_records_only",
        "neural_network": asdict(neural),
        "xgboost": asdict(xgboost),
        "cross_fit_folds": rule["cross_fit_folds"],
        "cross_fit_assignment": rule["cross_fit_assignment"],
        "gate_fit_partition": rule["gate_fit_partition"],
        "gate_features": tuple(rule["gate_features"]),
        "gate_feature_transformation_version": rule[
            "gate_feature_transformation_version"
        ],
        "ridge_penalty": rule["ridge_penalty"],
        "weight_constraint": rule["weight_constraint"],
    }
    digest = sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:8]
    return Candidate(
        f"gate-{index + 1:02d}-{digest}", "geometry_gate", parameters,
        CROSS_FITTED_GEOMETRY_GATE_FEATURES,
        CROSS_FITTED_GEOMETRY_GATE_TRANSFORMATION_VERSION,
    )


def _explicit_ensemble_candidate(
    index: int, neural: Candidate, xgboost: Candidate, weights: Sequence[float],
) -> Candidate:
    if (
        neural.ordered_prediction_features != xgboost.ordered_prediction_features
        or neural.feature_transformation_version
        != xgboost.feature_transformation_version
    ):
        raise ValueError("invalid_candidate_feature_contract")
    parameters = {
        "component_rank": index + 1,
        "selection_partition": "validation_records_only",
        "neural_network": asdict(neural),
        "xgboost": asdict(xgboost),
        "neural_network_weight_grid": tuple(weights),
    }
    digest = sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()[:8]
    return Candidate(
        f"ens-{index + 1:02d}-{digest}", "ensemble", parameters,
        neural.ordered_prediction_features, neural.feature_transformation_version,
    )


def candidate_prediction_matrix(
    rows: Sequence[CanonicalRow], candidate: Candidate,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Build the matrix bound to a candidate without exposing evaluation metadata."""
    if candidate.family == "control" or candidate.ordered_prediction_features == LEGACY_FEATURES:
        return _matrix(rows)
    if (
        candidate.ordered_prediction_features == LEGACY_GEOMETRY_AUGMENTATION_FEATURES
        and candidate.feature_transformation_version
        == LEGACY_GEOMETRY_AUGMENTATION_TRANSFORMATION_VERSION
    ):
        legacy, targets = _matrix(rows)
        additions = tuple(_legacy_geometry_augmentation_row(row) for row in rows)
        features = tuple(left + right for left, right in zip(legacy, additions))
        if len(legacy) != len(additions) or not _valid_float32_matrix(features):
            raise ValueError("invalid_candidate_feature_data")
        return features, targets
    if (
        candidate.ordered_prediction_features == CROSS_FITTED_GEOMETRY_GATE_FEATURES
        and candidate.feature_transformation_version
        == CROSS_FITTED_GEOMETRY_GATE_TRANSFORMATION_VERSION
        and candidate.family == "geometry_gate"
    ):
        legacy, targets = _matrix(rows)
        geometry = tuple(_geometry_regime_row(row) for row in rows)
        if len(legacy) != len(geometry):
            raise ValueError("invalid_candidate_feature_data")
        return tuple(left + right for left, right in zip(legacy, geometry)), targets
    if (
        candidate.ordered_prediction_features != GEOMETRY_REGIME_FEATURES
        or candidate.feature_transformation_version
        != GEOMETRY_REGIME_TRANSFORMATION_VERSION
    ):
        raise ValueError("invalid_candidate_feature_contract")
    features = tuple(_geometry_regime_row(row) for row in rows)
    targets = tuple(
        float(row.sliced_resin_mass_g) for row in rows
        if row.sliced_resin_mass_g is not None
    )
    if len(features) != len(targets) or not features:
        raise ValueError("invalid_candidate_feature_data")
    return features, targets


def _legacy_geometry_augmentation_row(row: CanonicalRow) -> tuple[float, ...]:
    try:
        selected = tuple(row.features[name] for name in GEOMETRY_REGIME_INPUTS)
        if any(value is None for value in selected):
            raise ValueError("missing geometry")
        raw = tuple(float(value) for value in selected if value is not None)
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid_candidate_feature_data") from None
    if not all(
        math.isfinite(value) and abs(value) <= FLOAT32_MAXIMUM for value in raw
    ):
        raise ValueError("invalid_candidate_feature_data")
    volume, surface, x, y, z, bbox_volume, _ = raw
    if any(value <= 0.0 for value in (volume, surface, x, y, z, bbox_volume)):
        raise ValueError("invalid_candidate_feature_data")
    short, _, long = sorted((x, y, z))
    values = (
        volume / bbox_volume,
        surface / bbox_volume,
        math.log1p(volume) ** 2,
        long / short,
    )
    if not all(
        math.isfinite(value) and abs(value) <= FLOAT32_MAXIMUM for value in values
    ):
        raise ValueError("invalid_candidate_feature_data")
    return values


def _geometry_regime_row(row: CanonicalRow) -> tuple[float, ...]:
    try:
        selected = tuple(row.features[name] for name in GEOMETRY_REGIME_INPUTS)
        if any(value is None for value in selected):
            raise ValueError("missing geometry")
        raw = tuple(float(value) for value in selected if value is not None)
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid_candidate_feature_data") from None
    if not all(math.isfinite(value) for value in raw):
        raise ValueError("invalid_candidate_feature_data")
    volume, surface, x, y, z, bbox_volume, euler = raw
    if any(value <= 0.0 for value in (volume, surface, x, y, z, bbox_volume)):
        raise ValueError("invalid_candidate_feature_data")
    short, middle, long = sorted((x, y, z))
    values = (
        volume, surface, short, middle, long, bbox_volume, euler,
        math.log1p(volume), math.log1p(surface), math.log1p(bbox_volume),
        math.log1p(short), math.log1p(middle), math.log1p(long),
        math.log(volume) - math.log(bbox_volume),
        math.log(surface) - math.log(volume),
        math.log(long) - math.log(short),
    )
    if not all(
        math.isfinite(value) and abs(value) <= FLOAT32_MAXIMUM for value in values
    ):
        raise ValueError("invalid_candidate_feature_data")
    return values


def _valid_float32_matrix(rows: Sequence[tuple[float, ...]]) -> bool:
    return bool(rows) and all(
        math.isfinite(value) and abs(value) <= FLOAT32_MAXIMUM
        for row in rows for value in row
    )


def _valid_candidate_feature_data(
    rows: Sequence[CanonicalRow], prediction_features: tuple[str, ...],
) -> bool:
    if prediction_features == LEGACY_FEATURES:
        return True
    if prediction_features not in {
        GEOMETRY_REGIME_FEATURES, CROSS_FITTED_GEOMETRY_GATE_FEATURES,
        LEGACY_GEOMETRY_AUGMENTATION_FEATURES,
    } or not rows:
        return False
    feature_builder = (
        _legacy_geometry_augmentation_row
        if prediction_features == LEGACY_GEOMETRY_AUGMENTATION_FEATURES
        else _geometry_regime_row
    )
    try:
        if not all(
            row.sliced_resin_mass_g is not None and bool(feature_builder(row))
            for row in rows
        ):
            return False
        if prediction_features == LEGACY_GEOMETRY_AUGMENTATION_FEATURES:
            legacy, _ = _matrix(rows)
            additions = tuple(_legacy_geometry_augmentation_row(row) for row in rows)
            return _valid_float32_matrix(tuple(
                left + right for left, right in zip(legacy, additions)
            ))
        return True
    except ValueError:
        return False


def _uses_tail_aligned_selection(candidate: Candidate) -> bool:
    if candidate.family in {"neural_network", "xgboost"}:
        return candidate.parameters.get("validation_selection") == (
            TAIL_ALIGNED_VALIDATION_SELECTION
        )
    if candidate.family == "ensemble":
        return all(
            _uses_tail_aligned_selection(_candidate_from_dict(candidate.parameters[name]))
            for name in ("neural_network", "xgboost")
        )
    return False


def _tail_prediction_ranker(
    training_count: int, validation: Sequence[CanonicalRow],
) -> Callable[[Sequence[float]], tuple[Any, ...]]:
    def rank(predictions: Sequence[float]) -> tuple[Any, ...]:
        values = tuple(float(value) for value in predictions)
        if len(values) != len(validation) or not all(map(math.isfinite, values)):
            return (2, math.inf, math.inf, math.inf, math.inf, 0.0)
        reports = _explicit_validation_source_reports(
            training_count, validation, values
        )
        return _tail_selection_key(_candidate_metrics(reports))
    return rank


def _fit_explicit_component(
    runtime: CandidateRuntime, candidate: Candidate, seed: int,
    train_x: Sequence[tuple[float, ...]], train_y: Sequence[float],
    validation_x: Sequence[tuple[float, ...]], validation_y: Sequence[float],
    prediction_ranker: Callable[[Sequence[float]], tuple[Any, ...]] | None,
) -> CandidateFoldFit:
    if prediction_ranker is None:
        return runtime.fit_fold(
            candidate, seed, train_x, train_y, validation_x, validation_y
        )
    return runtime.fit_fold(
        candidate, seed, train_x, train_y, validation_x, validation_y,
        prediction_ranker=prediction_ranker,
    )


def _evaluate_explicit_candidate(
    candidate: Candidate, seed: int, runtime: CandidateRuntime,
    training: Sequence[CanonicalRow], validation: Sequence[CanonicalRow],
    weight_grid: Sequence[float],
) -> CandidateRun:
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        if candidate.family == "geometry_gate":
            return _evaluate_explicit_geometry_gate(
                candidate, seed, runtime, training, validation, started, cpu_started
            )
        train_x, train_y = candidate_prediction_matrix(training, candidate)
        validation_x, validation_y = candidate_prediction_matrix(validation, candidate)
        prediction_ranker = (
            _tail_prediction_ranker(len(training), validation)
            if _uses_tail_aligned_selection(candidate) else None
        )
        if candidate.family == "ensemble":
            neural = _candidate_from_dict(candidate.parameters["neural_network"])
            xgboost = _candidate_from_dict(candidate.parameters["xgboost"])
            neural_fit = _fit_explicit_component(
                runtime, neural, seed, train_x, train_y, validation_x, validation_y,
                prediction_ranker,
            )
            xgboost_fit = _fit_explicit_component(
                runtime, xgboost, seed, train_x, train_y, validation_x, validation_y,
                prediction_ranker,
            )
            neural_predictions = _predict(neural_fit.predictor, validation_x)
            xgboost_predictions = _predict(xgboost_fit.predictor, validation_x)
            weight = _select_ensemble_weight(
                validation_y, neural_predictions, xgboost_predictions, weight_grid,
                prediction_ranker=prediction_ranker,
            )
            specification = ensemble_model_specification(
                neural.specification, xgboost.specification, weight
            )
            predictions = combine_ensemble_predictions(
                specification, neural_predictions, xgboost_predictions
            )
            metadata = {
                "neural_network": copy.deepcopy(neural_fit.metadata),
                "xgboost": copy.deepcopy(xgboost_fit.metadata),
                "selected_neural_network_weight": weight,
                "model_specification": specification.to_dict(),
                "fitted_state_fingerprint": fingerprint({
                    "neural_network": _fingerprintable_state(neural_fit.fitted_state),
                    "xgboost": _fingerprintable_state(xgboost_fit.fitted_state),
                }),
            }
        else:
            fitted = _fit_explicit_component(
                runtime, candidate, seed, train_x, train_y,
                validation_x, validation_y, prediction_ranker,
            )
            predictions = _predict(fitted.predictor, validation_x)
            metadata = {
                **copy.deepcopy(fitted.metadata),
                "fitted_state_fingerprint": fingerprint(
                    _fingerprintable_state(fitted.fitted_state)
                ),
            }
        reports = _explicit_validation_source_reports(
            len(training), validation, predictions
        )
        metrics = _candidate_metrics(reports)
        eligible = _tail_eligible(metrics)
        return CandidateRun(
            candidate, seed, "completed",
            () if eligible else ("development_serious_error_gate_failed",),
            eligible, metrics, reports, (metadata,),
            _resources(time.perf_counter() - started, time.process_time() - cpu_started),
        )
    except Exception:
        return CandidateRun(
            candidate, seed, "failed", ("candidate_runtime_failed",), False,
            _empty_metrics(), (), (),
            _resources(time.perf_counter() - started, time.process_time() - cpu_started),
        )


def _evaluate_explicit_geometry_gate(
    candidate: Candidate, seed: int, runtime: CandidateRuntime,
    training: Sequence[CanonicalRow], validation: Sequence[CanonicalRow],
    started: float, cpu_started: float,
) -> CandidateRun:
    _validate_geometry_gate_candidate(candidate)
    neural = _candidate_from_dict(candidate.parameters["neural_network"])
    xgboost = _candidate_from_dict(candidate.parameters["xgboost"])
    gate_state, fold_metadata = _fit_training_oof_geometry_gate(
        candidate, seed, runtime, training
    )
    fold_count = int(candidate.parameters["cross_fit_folds"])
    neural_train_x, neural_train_y = candidate_prediction_matrix(training, neural)
    neural_validation_x, validation_y = candidate_prediction_matrix(validation, neural)
    xgboost_train_x, xgboost_train_y = candidate_prediction_matrix(training, xgboost)
    xgboost_validation_x, xgboost_validation_y = candidate_prediction_matrix(
        validation, xgboost
    )
    if neural_train_y != xgboost_train_y or validation_y != xgboost_validation_y:
        raise ValueError("invalid_candidate_feature_data")
    neural_fit = runtime.fit_fold(
        neural, seed, neural_train_x, neural_train_y,
        neural_validation_x, validation_y,
    )
    xgboost_fit = runtime.fit_fold(
        xgboost, seed, xgboost_train_x, xgboost_train_y,
        xgboost_validation_x, validation_y,
    )
    neural_predictions = _predict(neural_fit.predictor, neural_validation_x)
    xgboost_predictions = _predict(xgboost_fit.predictor, xgboost_validation_x)
    validation_geometry = tuple(_geometry_regime_row(row) for row in validation)
    predictions = _geometry_gate_predictions(
        gate_state, validation_geometry, neural_predictions, xgboost_predictions
    )
    metadata = {
        "neural_network": copy.deepcopy(neural_fit.metadata),
        "xgboost": copy.deepcopy(xgboost_fit.metadata),
        "gate_training": {
            "partition": "training_oof_predictions_only",
            "cross_fit_folds": fold_count,
            "oof_prediction_count": len(training),
            "source_metadata_used": False,
        },
        "cross_fit_metadata": fold_metadata,
        "gate_state": gate_state,
        "fitted_state_fingerprint": fingerprint({
            "neural_network": _fingerprintable_state(neural_fit.fitted_state),
            "xgboost": _fingerprintable_state(xgboost_fit.fitted_state),
            "gate_state": gate_state,
        }),
    }
    reports = _explicit_validation_source_reports(
        len(training), validation, predictions
    )
    metrics = _candidate_metrics(reports)
    eligible = _tail_eligible(metrics)
    return CandidateRun(
        candidate, seed, "completed",
        () if eligible else ("development_serious_error_gate_failed",),
        eligible, metrics, reports, (metadata,),
        _resources(time.perf_counter() - started, time.process_time() - cpu_started),
    )


def _fit_training_oof_geometry_gate(
    candidate: Candidate, seed: int, runtime: CandidateRuntime,
    training: Sequence[CanonicalRow],
    fixed_training_counts: Mapping[str, int | float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    neural = _candidate_from_dict(candidate.parameters["neural_network"])
    xgboost = _candidate_from_dict(candidate.parameters["xgboost"])
    fold_count = int(candidate.parameters["cross_fit_folds"])
    fold_assignments = _cross_fit_assignments(training, seed, fold_count)
    oof_neural = [math.nan] * len(training)
    oof_xgboost = [math.nan] * len(training)
    fold_metadata: list[dict[str, Any]] = []
    for fold in range(fold_count):
        held_indices = [index for index, assigned in enumerate(fold_assignments)
                        if assigned == fold]
        fit_indices = [index for index, assigned in enumerate(fold_assignments)
                       if assigned != fold]
        if not held_indices or not fit_indices:
            raise ValueError("invalid_cross_fit_partition")
        fold_training = [training[index] for index in fit_indices]
        fold_validation = [training[index] for index in held_indices]
        train_x, train_y = candidate_prediction_matrix(fold_training, neural)
        held_x, held_y = candidate_prediction_matrix(fold_validation, neural)
        if fixed_training_counts is None:
            neural_fit = runtime.fit_fold(
                neural, seed, train_x, train_y, held_x, held_y
            )
            xgboost_fit = runtime.fit_fold(
                xgboost, seed, train_x, train_y, held_x, held_y
            )
        else:
            neural_locked = runtime.refit(
                neural, seed, train_x, train_y,
                {"neural_network_epochs": fixed_training_counts[
                    "neural_network_epochs"
                ]},
            )
            xgboost_locked = runtime.refit(
                xgboost, seed, train_x, train_y,
                {"xgboost_trees": fixed_training_counts["xgboost_trees"]},
            )
            neural_fit = CandidateFoldFit(
                neural_locked.predictor, dict(neural_locked.metadata), {}
            )
            xgboost_fit = CandidateFoldFit(
                xgboost_locked.predictor, dict(xgboost_locked.metadata), {}
            )
        for index, prediction in zip(
            held_indices, _predict(neural_fit.predictor, held_x)
        ):
            oof_neural[index] = prediction
        for index, prediction in zip(
            held_indices, _predict(xgboost_fit.predictor, held_x)
        ):
            oof_xgboost[index] = prediction
        fold_metadata.append({
            "fold": fold,
            "training_record_count": len(fit_indices),
            "held_out_record_count": len(held_indices),
            "neural_network": copy.deepcopy(neural_fit.metadata),
            "xgboost": copy.deepcopy(xgboost_fit.metadata),
        })
    _, train_y = candidate_prediction_matrix(training, neural)
    gate_features = tuple(_geometry_regime_row(row) for row in training)
    gate_state = _fit_geometry_gate(
        gate_features, train_y, oof_neural, oof_xgboost,
        float(candidate.parameters["ridge_penalty"]),
    )
    return gate_state, fold_metadata


def _cross_fit_assignments(
    rows: Sequence[CanonicalRow], seed: int, fold_count: int,
) -> tuple[int, ...]:
    if len(rows) < fold_count or fold_count < 2:
        raise ValueError("invalid_cross_fit_partition")
    identities = []
    for index, row in enumerate(rows):
        identity = row.metadata.get("record_identity")
        if not isinstance(identity, str) or not identity:
            raise ValueError("invalid_cross_fit_partition")
        digest = sha256(f"{seed}:{identity}".encode()).hexdigest()
        identities.append((digest, identity, index))
    if len({identity for _, identity, _ in identities}) != len(rows):
        raise ValueError("invalid_cross_fit_partition")
    assignments = [0] * len(rows)
    for position, (_, _, index) in enumerate(sorted(identities)):
        assignments[index] = position % fold_count
    return tuple(assignments)


def _fit_geometry_gate(
    geometry: Sequence[tuple[float, ...]], targets: Sequence[float],
    neural: Sequence[float], xgboost: Sequence[float], ridge_penalty: float,
) -> dict[str, Any]:
    import numpy as np

    if (
        not geometry or len(geometry) != len(targets)
        or len(targets) != len(neural) or len(targets) != len(xgboost)
        or not math.isfinite(ridge_penalty) or ridge_penalty <= 0.0
    ):
        raise ValueError("invalid_geometry_gate_training_data")
    values = np.asarray(geometry, dtype=np.float64)
    actual = np.asarray(targets, dtype=np.float64)
    left = np.asarray(neural, dtype=np.float64)
    right = np.asarray(xgboost, dtype=np.float64)
    if not all(np.isfinite(item).all() for item in (values, actual, left, right)):
        raise ValueError("invalid_geometry_gate_training_data")
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (values - means) / scales
    basis = np.column_stack((np.ones(len(values)), standardized))
    delta = left - right
    design = basis * delta[:, None]
    response = actual - right
    penalty = np.sqrt(ridge_penalty) * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    augmented_design = np.vstack((design, penalty))
    augmented_response = np.concatenate((response, np.zeros(design.shape[1])))
    coefficients, *_ = np.linalg.lstsq(
        augmented_design, augmented_response, rcond=None
    )
    state = {
        "version": CROSS_FITTED_GEOMETRY_GATE_TRANSFORMATION_VERSION,
        "gate_features": list(GEOMETRY_REGIME_FEATURES),
        "weight_constraint": "clip_0_1",
        "ridge_penalty": ridge_penalty,
        "means": means.tolist(),
        "scales": scales.tolist(),
        "coefficients": coefficients.tolist(),
    }
    if not _valid_geometry_gate_state(state):
        raise ValueError("invalid_geometry_gate_training_data")
    return state


def _geometry_gate_predictions(
    state: Mapping[str, Any], geometry: Sequence[tuple[float, ...]],
    neural: Sequence[float], xgboost: Sequence[float],
) -> tuple[float, ...]:
    import numpy as np

    if not _valid_geometry_gate_state(state) or len(geometry) != len(neural) \
            or len(neural) != len(xgboost):
        raise ValueError("invalid_geometry_gate_state")
    values = np.asarray(geometry, dtype=np.float64)
    means = np.asarray(state["means"], dtype=np.float64)
    scales = np.asarray(state["scales"], dtype=np.float64)
    coefficients = np.asarray(state["coefficients"], dtype=np.float64)
    basis = np.column_stack((np.ones(len(values)), (values - means) / scales))
    weights = np.clip(basis @ coefficients, 0.0, 1.0)
    predictions = tuple(
        float(weight) * float(left) + (1.0 - float(weight)) * float(right)
        for weight, left, right in zip(weights, neural, xgboost)
    )
    if not all(math.isfinite(value) for value in predictions):
        raise ValueError("invalid_model_prediction")
    return predictions


def _valid_geometry_gate_state(state: Mapping[str, Any]) -> bool:
    width = len(GEOMETRY_REGIME_FEATURES)
    return (
        state.get("version") == CROSS_FITTED_GEOMETRY_GATE_TRANSFORMATION_VERSION
        and state.get("gate_features") == list(GEOMETRY_REGIME_FEATURES)
        and state.get("weight_constraint") == "clip_0_1"
        and isinstance(state.get("ridge_penalty"), (int, float))
        and not isinstance(state.get("ridge_penalty"), bool)
        and math.isfinite(float(state["ridge_penalty"]))
        and float(state["ridge_penalty"]) > 0.0
        and all(
            isinstance(state.get(key), list) and len(state[key]) == expected
            and all(isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in state[key])
            for key, expected in (("means", width), ("scales", width),
                                  ("coefficients", width + 1))
        )
        and all(float(value) > 0.0 for value in state["scales"])
    )


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
    plan_identities = _search_plan_identities(rows, config)
    selected_plan = plan or _generate_bound_search_plan(
        limits, input_fingerprint, runtime.dependency_versions, plan_identities,
    )
    try:
        _validate_plan(
            selected_plan, limits, input_fingerprint,
            runtime.dependency_versions, plan_identities,
        )
    except ValueError:
        evidence_plan = _generate_bound_search_plan(
            limits, input_fingerprint, runtime.dependency_versions, plan_identities,
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

    included_rows = tuple(row for row in rows if row.outcome == "included")
    if not _valid_candidate_feature_data(
        included_rows, selected_plan.prediction_features
    ):
        result = TuningResult(
            "blocked", ("invalid_candidate_feature_data",), 0,
            {"neural_network": 0, "xgboost": 0, "ensemble": 0,
             "second_seed": 0, "control": 0},
            selected_plan, None, (), (), (), None, _resources(0.0, 0.0),
        )
        _write_tuning_outputs(output, result)
        return result

    by_index = {row.row_index: row for row in included_rows}
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
    ranked_initial = sorted((item for item in initial if item.eligible), key=_rank_key)
    repetition_target = min(limits.second_seed_candidates, len(ranked_initial))
    initial_complete = len(initial) == (
        len(selected_plan.component_trials) + limits.ensemble_trials
    )
    if initial_complete and not blockers:
        for first in ranked_initial[:repetition_target]:
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
    complete = (
        initial_complete
        and len(second) == repetition_target
        and all(run.status == "completed" for run in second)
        and not blockers
    )
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
                    candidate, selected, selected_plan, runtime, tuple(by_index.values()),
                    initial, second, manifest, output / "locked-candidate",
                )
            except Exception:
                blockers.append("candidate_refit_or_lock_failed")
    elapsed = max(0.0, clock() - started)
    resources = _resources(elapsed, time.process_time() - cpu_started)
    status = (
        "completed" if locked is not None
        else "completed_no_candidate"
        if complete and "no_eligible_candidate" in blockers
        else "blocked"
    )
    result = TuningResult(
        status, tuple(sorted(set(blockers))), run_count, allocation, selected_plan,
        control_result, tuple(initial), tuple(second), tuple(combined), locked, resources,
    )
    _write_tuning_outputs(output, result)
    return result


class _CandidateArtifactError(Exception):
    pass


def _evaluate_candidate(
    candidate: Candidate, seed: int, runtime: CandidateRuntime,
    manifest: Mapping[str, Any], by_index: Mapping[int, CanonicalRow],
    weight_grid: Sequence[float], artifact_root: Path | None = None,
) -> CandidateRun:
    reports: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    artifact_checksums: dict[str, str] = {}
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        for fold_number, fold in enumerate(manifest["folds"]):
            train_x, train_y = candidate_prediction_matrix(
                [by_index[index] for index in fold["train"]], candidate
            )
            validation_x, validation_y = candidate_prediction_matrix(
                [by_index[index] for index in fold["validation"]], candidate
            )
            test_x, test_y = candidate_prediction_matrix(
                [by_index[index] for index in fold["test"]], candidate
            )
            if candidate.family == "ensemble":
                fold_components = next(
                    item for item in candidate.parameters["components_by_fold"]
                    if item["source"] == fold["source"]
                )
                neural = _candidate_from_dict(fold_components["neural_network"])
                xgboost = _candidate_from_dict(fold_components["xgboost"])
                neural_fit = runtime.fit_fold(neural, seed, train_x, train_y, validation_x, validation_y)
                xgboost_fit = runtime.fit_fold(xgboost, seed, train_x, train_y, validation_x, validation_y)
                neural_metadata = copy.deepcopy(neural_fit.metadata)
                xgboost_metadata = copy.deepcopy(xgboost_fit.metadata)
                if artifact_root is not None:
                    artifact_checksums.update(_store_fold_artifacts(
                        runtime, neural_fit, artifact_root, fold_number, "neural-network-"
                    ))
                    artifact_checksums.update(_store_fold_artifacts(
                        runtime, xgboost_fit, artifact_root, fold_number, "xgboost-"
                    ))
                validation_neural = _predict(neural_fit.predictor, validation_x)
                validation_xgboost = _predict(xgboost_fit.predictor, validation_x)
                weight = _select_ensemble_weight(validation_y, validation_neural,
                                                 validation_xgboost, weight_grid)
                ensemble_specification = ensemble_model_specification(
                    neural.specification, xgboost.specification, weight,
                )
                fitted_state_fingerprint = fingerprint({
                    "neural_network": _fingerprintable_state(neural_fit.fitted_state),
                    "xgboost": _fingerprintable_state(xgboost_fit.fitted_state),
                    "model_specification": ensemble_specification.to_dict(),
                })
                test_neural = _predict(neural_fit.predictor, test_x)
                test_xgboost = _predict(xgboost_fit.predictor, test_x)
                predictions = combine_ensemble_predictions(
                    ensemble_specification, test_neural, test_xgboost
                )
                fold_metadata = {
                    "neural_network": neural_metadata,
                    "xgboost": xgboost_metadata,
                    "selected_neural_network_weight": weight,
                    "model_specification": ensemble_specification.to_dict(),
                    "fitted_state_fingerprint": fitted_state_fingerprint,
                }
            else:
                fitted = runtime.fit_fold(candidate, seed, train_x, train_y,
                                          validation_x, validation_y)
                recorded_metadata = copy.deepcopy(fitted.metadata)
                validation_predictions = _predict(fitted.predictor, validation_x)
                fitted_state_fingerprint = fingerprint(
                    _fingerprintable_state(fitted.fitted_state)
                )
                if artifact_root is not None:
                    artifact_checksums.update(_store_fold_artifacts(
                        runtime, fitted, artifact_root, fold_number, ""
                    ))
                predictions = _predict(fitted.predictor, test_x)
                fold_metadata = {
                    **recorded_metadata,
                    "fitted_state_fingerprint": fitted_state_fingerprint,
                }
            source_metrics = _source_candidate_metrics(test_y, predictions)
            reports.append({
                "source": fold["source"],
                "test_rows": list(fold["test"]),
                "train_rows": list(fold["train"]),
                "validation_rows": list(fold["validation"]),
                "actual": list(test_y),
                "predictions": list(predictions),
                **source_metrics,
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
            artifact_checksums,
        )
    except _CandidateArtifactError:
        return CandidateRun(
            candidate, seed, "blocked", ("candidate_artifact_failed",), False,
            _empty_metrics(), (), (),
            _resources(time.perf_counter() - started, time.process_time() - cpu_started),
            artifact_checksums,
        )
    except Exception:
        return CandidateRun(
            candidate, seed, "failed", ("candidate_runtime_failed",), False,
            _empty_metrics(), (), (),
            _resources(time.perf_counter() - started, time.process_time() - cpu_started),
            artifact_checksums,
        )


def _store_fold_artifacts(
    runtime: CandidateRuntime, fitted: CandidateFoldFit, root: Path,
    fold_number: int, prefix: str,
) -> dict[str, str]:
    try:
        artifacts = runtime.serialize_fold(fitted)
        if not artifacts:
            raise ValueError("missing fitted artifacts")
        directory = root / f"fold-{fold_number:03d}"
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        checksums: dict[str, str] = {}
        for name, content in sorted(artifacts.items()):
            stored_name = f"{prefix}{name}"
            if (not name or Path(name).name != name or Path(stored_name).name != stored_name
                    or not isinstance(content, bytes)):
                raise ValueError("invalid fitted artifact")
            path = directory / stored_name
            with create_private_file(path) as stream:
                stream.write(content)
            checksums[str(path.relative_to(root.parent))] = sha256(content).hexdigest()
        return checksums
    except (AttributeError, InputError, OSError, TypeError, ValueError):
        raise _CandidateArtifactError from None


def _fingerprintable_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if not key.startswith("_")}


def _explicit_validation_source_reports(
    training_count: int, validation: Sequence[CanonicalRow],
    predictions: Sequence[float],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, dict[str, Any]] = {}
    for row, prediction in zip(validation, predictions, strict=True):
        target = row.sliced_resin_mass_g
        if target is None:
            raise ValueError("validation target required")
        source = str(row.metadata["anonymous_source_group"])
        report = grouped.setdefault(source, {
            "source": source,
            "train_rows": list(range(training_count)),
            "validation_rows": [],
            "actual": [],
            "predictions": [],
        })
        report["validation_rows"].append(row.row_index)
        report["actual"].append(float(target))
        report["predictions"].append(float(prediction))
    reports = []
    for source in sorted(grouped):
        report = grouped[source]
        report.update(_source_candidate_metrics(report["actual"], report["predictions"]))
        reports.append(report)
    return tuple(reports)


def _source_candidate_metrics(
    actual: Sequence[float], predictions: Sequence[float]
) -> dict[str, float | int]:
    if len(actual) != len(predictions) or not all(
        math.isfinite(float(value)) for value in (*actual, *predictions)
    ):
        raise ValueError("invalid_candidate_predictions")
    errors = [abs(float(prediction) - float(target))
              for prediction, target in zip(predictions, actual)]
    return {
        "sample_count": len(errors),
        "mae_g": math.fsum(errors) / len(errors),
        "within_2g_fraction": sum(error <= 2.0 for error in errors) / len(errors),
        "above_5g_fraction": sum(error > 5.0 for error in errors) / len(errors),
    }


def _candidate_metrics(reports: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    source_metrics = []
    all_errors: list[float] = []
    for report in reports:
        errors = [abs(float(prediction) - float(actual))
                  for prediction, actual in zip(report["predictions"], report["actual"])]
        all_errors.extend(errors)
        source_metrics.append({
            "sample_count": report["sample_count"],
            "mae": report["mae_g"],
            "within": report["within_2g_fraction"],
            "tail": report["above_5g_fraction"],
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
        "qualifying_source_count": sum(
            item["sample_count"] >= PER_SOURCE_MINIMUM_ACCEPTED_RECORDS
            for item in source_metrics
        ),
        "maximum_qualifying_source_above_5g_fraction": max(
            (item["tail"] for item in source_metrics
             if item["sample_count"] >= PER_SOURCE_MINIMUM_ACCEPTED_RECORDS),
            default=None,
        ),
    }


def _empty_metrics() -> dict[str, float | int | None]:
    return {key: 0 if key in {"sample_count", "source_count"} else None for key in (
        "sample_count", "source_count", "pooled_mae_g", "source_balanced_mae_g",
        "pooled_within_2g_fraction", "source_balanced_within_2g_fraction",
        "pooled_above_5g_fraction", "source_balanced_above_5g_fraction",
        "maximum_source_above_5g_fraction", "qualifying_source_count",
        "maximum_qualifying_source_above_5g_fraction",
    )}


def _eligibility_gates() -> dict[str, float | int]:
    return {
        "pooled_above_5g_fraction_maximum": POOLED_ABOVE_5G_FRACTION_MAXIMUM,
        "source_balanced_above_5g_fraction_maximum": (
            SOURCE_BALANCED_ABOVE_5G_FRACTION_MAXIMUM
        ),
        "per_source_above_5g_fraction_maximum": PER_SOURCE_ABOVE_5G_FRACTION_MAXIMUM,
        "per_source_minimum_accepted_records": PER_SOURCE_MINIMUM_ACCEPTED_RECORDS,
    }


def _tail_eligible(metrics: Mapping[str, Any]) -> bool:
    return (
        metrics["pooled_above_5g_fraction"] is not None
        and metrics["pooled_above_5g_fraction"] <= POOLED_ABOVE_5G_FRACTION_MAXIMUM
        and metrics["source_balanced_above_5g_fraction"]
        <= SOURCE_BALANCED_ABOVE_5G_FRACTION_MAXIMUM
        and (
            metrics["maximum_qualifying_source_above_5g_fraction"] is None
            or metrics["maximum_qualifying_source_above_5g_fraction"]
            <= PER_SOURCE_ABOVE_5G_FRACTION_MAXIMUM
        )
    )


def _tail_selection_key(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    pooled = metrics.get("pooled_above_5g_fraction")
    balanced = metrics.get("source_balanced_above_5g_fraction")
    maximum = metrics.get("maximum_qualifying_source_above_5g_fraction")
    ranked_values = (
        pooled, balanced, maximum,
        metrics.get("source_balanced_mae_g"), metrics.get("pooled_mae_g"),
        metrics.get("pooled_within_2g_fraction"),
    )
    if (
        pooled is None or balanced is None
        or any(value is not None and not math.isfinite(float(value))
               for value in ranked_values)
    ):
        return (2, math.inf, math.inf, math.inf, math.inf, 0.0)
    ratios = (
        float(pooled) / POOLED_ABOVE_5G_FRACTION_MAXIMUM,
        float(balanced) / SOURCE_BALANCED_ABOVE_5G_FRACTION_MAXIMUM,
        0.0 if maximum is None else (
            float(maximum) / PER_SOURCE_ABOVE_5G_FRACTION_MAXIMUM
        ),
    )
    excesses = tuple(max(0.0, ratio - 1.0) for ratio in ratios)
    return (
        0 if _tail_eligible(metrics) else 1,
        max(excesses),
        math.fsum(excesses),
        metrics.get("source_balanced_mae_g", math.inf),
        metrics.get("pooled_mae_g", math.inf),
        -(metrics.get("pooled_within_2g_fraction") or 0.0),
    )


def _tail_component_rank_key(run: CandidateRun) -> tuple[Any, ...]:
    return (*_tail_selection_key(run.metrics), run.candidate.candidate_id)


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
    features = neural[0].candidate.ordered_prediction_features
    transformation = neural[0].candidate.feature_transformation_version
    if any(
        run.candidate.ordered_prediction_features != features
        or run.candidate.feature_transformation_version != transformation
        for run in (*neural, *xgboost)
    ):
        raise ValueError("invalid_candidate_feature_contract")
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
    return Candidate(
        f"ens-{index + 1:02d}-{digest}", "ensemble", parameters, features,
        neural[0].candidate.feature_transformation_version,
    )


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
        "development_components_by_fold": candidate.parameters["components_by_fold"],
        "neural_network_weight_grid": candidate.parameters["neural_network_weight_grid"],
        "component_resolution": "most_frequent_fold_validation_pair_then_stable_id",
    }, candidate.ordered_prediction_features, candidate.feature_transformation_version)


def _candidate_from_dict(value: Mapping[str, Any]) -> Candidate:
    parameters = _freeze_json_lists(value["parameters"])
    features = _freeze_json_lists(value.get("ordered_prediction_features", LEGACY_FEATURES))
    transformation = value.get("feature_transformation_version", TRANSFORMATION_VERSION)
    if (
        not isinstance(parameters, dict) or not isinstance(features, tuple)
        or not isinstance(transformation, str) or not transformation
    ):
        raise ValueError("invalid candidate parameters")
    return Candidate(
        str(value["candidate_id"]), str(value["family"]), parameters, features,
        transformation,
    )


def _freeze_json_lists(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze_json_lists(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_json_lists(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _freeze_json_lists(item) for key, item in value.items()}
    return value


def _select_ensemble_weight(
    actual: Sequence[float], neural: Sequence[float], xgboost: Sequence[float],
    weights: Sequence[float], *,
    prediction_ranker: Callable[[Sequence[float]], tuple[Any, ...]] | None = None,
) -> float:
    def key(weight: float) -> tuple[Any, ...]:
        predictions = tuple(
            weight * left + (1 - weight) * right
            for left, right in zip(neural, xgboost)
        )
        if prediction_ranker is not None:
            return (*prediction_ranker(predictions), weight)
        return (
            math.fsum(abs(prediction - target)
                      for prediction, target in zip(predictions, actual)) / len(actual),
            weight,
        )
    return min(weights, key=key)


def _combine_seed_results(initial: Sequence[CandidateRun], second: Sequence[CandidateRun]
                          ) -> list[dict[str, Any]]:
    by_id = {item.candidate.candidate_id: item for item in initial}
    combined = []
    for repeated in second:
        first = by_id[repeated.candidate.candidate_id]
        metrics = _equal_seed_metrics(first, repeated)
        seed_eligibility = [first.eligible, repeated.eligible]
        eligible = all(seed_eligibility) and _tail_eligible(metrics)
        combined.append({
            "candidate_id": first.candidate.candidate_id,
            "family": first.candidate.family,
            "eligible": eligible,
            "blockers": [] if eligible else ["development_serious_error_gate_failed"],
            "seed_results": [first.seed, repeated.seed],
            "seed_eligibility": seed_eligibility,
            "equal_seed_weight": 0.5,
            "metrics": metrics,
        })
    return combined


def _equal_seed_metrics(
    first: CandidateRun, repeated: CandidateRun,
) -> dict[str, float | int | None]:
    count_keys = {"sample_count", "source_count", "qualifying_source_count"}
    maximum_keys = {
        "maximum_source_above_5g_fraction",
        "maximum_qualifying_source_above_5g_fraction",
    }
    metrics: dict[str, float | int | None] = {}
    for key in first.metrics:
        left = first.metrics[key]
        right = repeated.metrics[key]
        if key in count_keys:
            metrics[key] = left if left == right and isinstance(left, int) else None
        elif key not in maximum_keys:
            metrics[key] = (
                (float(left) + float(right)) / 2
                if isinstance(left, (int, float)) and isinstance(right, (int, float))
                else None
            )

    first_sources = {report["source"]: report for report in first.source_reports}
    second_sources = {report["source"]: report for report in repeated.source_reports}
    if set(first_sources) != set(second_sources):
        metrics["maximum_source_above_5g_fraction"] = None
        metrics["maximum_qualifying_source_above_5g_fraction"] = None
        return metrics
    combined_sources = [
        {
            "sample_count": first_sources[source]["sample_count"],
            "tail": (
                float(first_sources[source]["above_5g_fraction"])
                + float(second_sources[source]["above_5g_fraction"])
            ) / 2,
        }
        for source in sorted(first_sources)
        if first_sources[source]["sample_count"] == second_sources[source]["sample_count"]
    ]
    if len(combined_sources) != len(first_sources):
        metrics["maximum_source_above_5g_fraction"] = None
        metrics["maximum_qualifying_source_above_5g_fraction"] = None
        return metrics
    metrics["maximum_source_above_5g_fraction"] = max(
        (source["tail"] for source in combined_sources), default=None
    )
    metrics["maximum_qualifying_source_above_5g_fraction"] = max(
        (source["tail"] for source in combined_sources
         if source["sample_count"] >= PER_SOURCE_MINIMUM_ACCEPTED_RECORDS),
        default=None,
    )
    return metrics


def _combined_rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = item["metrics"]
    return (metrics["source_balanced_mae_g"], metrics["pooled_mae_g"],
            -metrics["pooled_within_2g_fraction"], item["candidate_id"])


def _refit_and_lock_explicit(
    candidate: Candidate, selected_evidence: Mapping[str, Any], plan: SearchPlan,
    runtime: CandidateRuntime, training: Sequence[CanonicalRow],
    validation: Sequence[CanonicalRow], initial: Sequence[CandidateRun],
    second: Sequence[CandidateRun], training_fingerprint: str,
    validation_fingerprint: str, directory: Path,
) -> LockedCandidate:
    related = [run for run in (*initial, *second)
               if run.candidate.candidate_id == candidate.candidate_id]
    fixed_counts = _derive_training_counts(candidate, related)
    if not _valid_fixed_training_counts(candidate, fixed_counts):
        raise InputError("invalid_locked_candidate_training_counts")
    features, targets = candidate_prediction_matrix(training, candidate)
    fitted = (
        _refit_geometry_gate_candidate(
            candidate, plan.second_seed, runtime, training, fixed_counts
        )
        if candidate.family == "geometry_gate"
        else runtime.refit(candidate, plan.second_seed, features, targets, fixed_counts)
    )
    if not isinstance(fitted.preprocessing_state, Mapping) or not fitted.artifacts:
        raise InputError("invalid_locked_candidate_artifact")
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    model_specification = _locked_model_specification(candidate, fixed_counts)
    contract = {
        "version": TUNING_VERSION,
        "development_contract": "explicit_train_validation",
        "candidate": asdict(candidate),
        "model_specification": model_specification.to_dict(),
        "runtime_configuration": _locked_runtime_configuration(candidate),
        "selection_seeds": [plan.seed, plan.second_seed],
        "seed_weighting": "equal_weight_each_seed",
        "selected_combined_development_evidence": copy.deepcopy(dict(selected_evidence)),
        "development_evidence": {
            "search_plan_id": plan.plan_id,
            "training_input_fingerprint": training_fingerprint,
            "validation_input_fingerprint": validation_fingerprint,
            "normalized_training_fingerprint": fingerprint([asdict(row) for row in training]),
            "normalized_validation_fingerprint": fingerprint([asdict(row) for row in validation]),
            "partition_identity_fingerprint": plan.source_allocation_fingerprint,
            "search_plan_fingerprint": fingerprint(plan.to_dict()),
            "test_input_attestation": "no_test_argument_or_path_available",
            "validation_grouping_contract": EXPLICIT_DEVELOPMENT_EVIDENCE_VERSION,
        },
        "development_source_groups": sorted({
            str(row.metadata["anonymous_source_group"])
            for row in (*training, *validation)
        }),
        "development_data_usage": {
            "fitting": "training_records_only",
            "preprocessing": "training_records_only",
            "early_stopping": "validation_records_only",
            "ensemble_selection": "validation_records_only",
            "threshold_selection": "validation_records_only",
            "candidate_selection": "validation_records_only",
            "candidate_locking": "training_and_validation_contract_only",
            **({"gate_fitting": "training_oof_predictions_only"}
               if candidate.family == "geometry_gate" else {}),
        },
        "code_fingerprint": plan.code_fingerprint,
        "transformation_version": TRANSFORMATION_VERSION,
        "feature_contract": {
            "ordered_features": list(candidate.ordered_prediction_features),
            "transformation_version": candidate.feature_transformation_version,
            "dtype": "float32",
            "excluded_fields": [
                "anonymous_source_group", "miniature_family", "partition",
                "duplicate_group", "geometry_fingerprint", "record_identity",
                "location_evidence", "_id", "join_key",
            ],
        },
        "features": list(candidate.ordered_prediction_features),
        "preprocessing": "locked_fit_on_training_records_only",
        "preprocessing_state_file": "preprocessing-state.json",
        "eligibility_rule": _eligibility_gates(),
        "ranking_rule": list(plan.ranking_rule),
        "dependency_versions": dict(plan.dependency_versions),
        "dependency_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": dict(plan.dependency_versions),
        },
        "fixed_training_counts": fixed_counts,
        "training_count_rule": "selected_from_validation_evidence_across_both_seeds",
        "refit_partition": "training_records_only",
        "refit_record_count": len(training),
        "validation_record_count": len(validation),
        "test_input_accessed": False,
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
        "create_only": True,
        "locked_before_final_assessment": True,
        "test_input_accessed": False,
    }
    write_private_json(directory / "lock-manifest.json", manifest)
    return LockedCandidate(candidate, fitted.predictor, directory, manifest, contract)


def _refit_geometry_gate_candidate(
    candidate: Candidate, seed: int, runtime: CandidateRuntime,
    training: Sequence[CanonicalRow], fixed_counts: Mapping[str, int | float],
) -> LockedFit:
    fixed_neural, fixed_xgboost = _fixed_geometry_gate_components(
        candidate, fixed_counts
    )
    fixed_candidate = Candidate(
        candidate.candidate_id, candidate.family,
        {
            **candidate.parameters,
            "neural_network": asdict(fixed_neural),
            "xgboost": asdict(fixed_xgboost),
        },
        candidate.ordered_prediction_features,
        candidate.feature_transformation_version,
    )
    gate_state, fold_metadata = _fit_training_oof_geometry_gate(
        fixed_candidate, seed, runtime, training, fixed_counts
    )
    neural_x, targets = candidate_prediction_matrix(training, fixed_neural)
    xgboost_x, xgboost_targets = candidate_prediction_matrix(training, fixed_xgboost)
    if targets != xgboost_targets:
        raise ValueError("invalid_candidate_feature_data")
    neural_fit = runtime.refit(
        fixed_neural, seed, neural_x, targets,
        {"neural_network_epochs": fixed_counts["neural_network_epochs"]},
    )
    xgboost_fit = runtime.refit(
        fixed_xgboost, seed, xgboost_x, targets,
        {"xgboost_trees": fixed_counts["xgboost_trees"]},
    )

    predictor = _geometry_gate_predictor(
        gate_state, neural_fit.predictor, xgboost_fit.predictor
    )

    artifacts = {
        **{f"neural-{name}": content for name, content in neural_fit.artifacts.items()},
        **{f"xgboost-{name}": content for name, content in xgboost_fit.artifacts.items()},
        "gate-state.json": (
            json.dumps(gate_state, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n"
        ).encode(),
    }
    return LockedFit(
        predictor,
        {
            "neural_network": dict(neural_fit.preprocessing_state),
            "xgboost": dict(xgboost_fit.preprocessing_state),
            "gate": gate_state,
        },
        artifacts,
        {
            "seed": seed,
            "gate_training": "training_oof_predictions_only",
            "cross_fit_metadata": fold_metadata,
            "neural_network": dict(neural_fit.metadata),
            "xgboost": dict(xgboost_fit.metadata),
        },
    )


def _fixed_geometry_gate_components(
    candidate: Candidate, fixed_counts: Mapping[str, int | float],
) -> tuple[Candidate, Candidate]:
    neural = _candidate_from_dict(candidate.parameters["neural_network"])
    xgboost = _candidate_from_dict(candidate.parameters["xgboost"])
    neural_parameters = dict(neural.parameters)
    neural_parameters["maximum_epochs"] = fixed_counts["neural_network_epochs"]
    fixed_neural = Candidate(
        neural.candidate_id, neural.family, neural_parameters,
        neural.ordered_prediction_features, neural.feature_transformation_version,
    )
    xgboost_parameters = dict(xgboost.parameters)
    xgboost_parameters["n_estimators"] = fixed_counts["xgboost_trees"]
    fixed_xgboost = Candidate(
        xgboost.candidate_id, xgboost.family, xgboost_parameters,
        xgboost.ordered_prediction_features, xgboost.feature_transformation_version,
    )
    return fixed_neural, fixed_xgboost


def _geometry_gate_predictor(
    gate_state: Mapping[str, Any],
    neural_predictor: Callable[[Sequence[tuple[float, ...]]], Sequence[float]],
    xgboost_predictor: Callable[[Sequence[tuple[float, ...]]], Sequence[float]],
) -> Callable[[Sequence[tuple[float, ...]]], Sequence[float]]:
    def predict(rows: Sequence[tuple[float, ...]]) -> Sequence[float]:
        legacy = tuple(tuple(row[:len(LEGACY_FEATURES)]) for row in rows)
        geometry = tuple(tuple(row[len(LEGACY_FEATURES):]) for row in rows)
        return _geometry_gate_predictions(
            gate_state, geometry, neural_predictor(legacy),
            xgboost_predictor(legacy),
        )

    return predict


def _refit_and_lock(
    candidate: Candidate, selected_evidence: Mapping[str, Any], plan: SearchPlan,
    runtime: CandidateRuntime, rows: Sequence[CanonicalRow],
    initial: Sequence[CandidateRun], second: Sequence[CandidateRun],
    development_manifest: Mapping[str, Any], directory: Path,
) -> LockedCandidate:
    related = [run for run in (*initial, *second)
               if run.candidate.candidate_id == candidate.candidate_id]
    fixed_counts = _derive_training_counts(candidate, related)
    if not _valid_fixed_training_counts(candidate, fixed_counts):
        raise InputError("invalid_locked_candidate_training_counts")
    features, targets = candidate_prediction_matrix(rows, candidate)
    fitted = runtime.refit(candidate, plan.second_seed, features, targets, fixed_counts)
    if not isinstance(fitted.preprocessing_state, Mapping) or not fitted.artifacts:
        raise InputError("invalid_locked_candidate_artifact")
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    model_specification = _locked_model_specification(candidate, fixed_counts)
    contract = {
        "version": TUNING_VERSION,
        "candidate": asdict(candidate),
        "model_specification": model_specification.to_dict(),
        "runtime_configuration": _locked_runtime_configuration(candidate),
        "selection_seeds": [plan.seed, plan.second_seed],
        "seed_weighting": "equal_weight_each_seed",
        "selected_combined_development_evidence": copy.deepcopy(dict(selected_evidence)),
        "development_evidence": {
            "search_plan_id": plan.plan_id,
            "input_fingerprint": plan.input_fingerprint,
            "normalized_input_fingerprint": plan.normalized_input_fingerprint,
            "source_allocation_fingerprint": plan.source_allocation_fingerprint,
            "search_plan_fingerprint": fingerprint(plan.to_dict()),
            "development_split_fingerprint": fingerprint(development_manifest),
        },
        "development_source_groups": sorted({
            row.metadata["anonymous_source_group"] for row in rows
            if row.metadata.get("anonymous_source_group")
        }),
        "development_data_usage": {
            stage: "development_records_only" for stage in DEVELOPMENT_ONLY_STAGES
        },
        "code_fingerprint": plan.code_fingerprint,
        "transformation_version": TRANSFORMATION_VERSION,
        "feature_contract": {
            "ordered_features": list(candidate.ordered_prediction_features),
            "transformation_version": candidate.feature_transformation_version,
            "dtype": "float32",
        },
        "features": list(candidate.ordered_prediction_features),
        "preprocessing": "locked_fit_on_all_included_development_records_only",
        "preprocessing_state_file": "preprocessing-state.json",
        "eligibility_rule": _eligibility_gates(),
        "ranking_rule": list(plan.ranking_rule),
        "dependency_versions": dict(plan.dependency_versions),
        "dependency_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": dict(plan.dependency_versions),
        },
        "fixed_training_counts": fixed_counts,
        "training_count_rule": "median_across_permitted_development_folds_and_both_seeds",
        "refit_partition": "all_included_development_records",
        "refit_record_count": len(rows),
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
        "create_only": True,
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
            values = (
                metadata.values()
                if candidate.family in {"ensemble", "geometry_gate"}
                else (metadata,)
            )
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
        result["neural_network_epochs"] = _rounded_median_count(epochs)
    if trees:
        result["xgboost_trees"] = _rounded_median_count(trees)
    if weights:
        result["ensemble_neural_network_weight"] = statistics.median(weights)
    return result


def _rounded_median_count(values: Sequence[int]) -> int:
    return int(math.floor(float(statistics.median(values)) + 0.5))


def _locked_runtime_configuration(candidate: Candidate) -> dict[str, Any]:
    if candidate.family == "geometry_gate":
        return {
            "neural_network": copy.deepcopy(CANDIDATE_RUNTIME_CONTRACT["neural_network"]),
            "xgboost": copy.deepcopy(CANDIDATE_RUNTIME_CONTRACT["xgboost"]),
            "combination": "geometry_conditioned_clipped_weight",
            "gate_fit_partition": "training_oof_predictions_only",
            "cross_fit_folds": candidate.parameters.get("cross_fit_folds"),
            "gate_transformation_version": candidate.feature_transformation_version,
        }
    if candidate.family == "ensemble":
        return {
            "neural_network": copy.deepcopy(CANDIDATE_RUNTIME_CONTRACT["neural_network"]),
            "xgboost": copy.deepcopy(CANDIDATE_RUNTIME_CONTRACT["xgboost"]),
            "combination": "locked_convex_weight",
        }
    return copy.deepcopy(CANDIDATE_RUNTIME_CONTRACT[candidate.family])


def _valid_fixed_training_counts(
    candidate: Candidate, fixed_counts: Mapping[str, Any],
) -> bool:
    epochs = fixed_counts.get("neural_network_epochs")
    trees = fixed_counts.get("xgboost_trees")
    weight = fixed_counts.get("ensemble_neural_network_weight")
    return (
        candidate.family == "neural_network"
        and isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0
        or candidate.family == "xgboost"
        and isinstance(trees, int) and not isinstance(trees, bool) and trees > 0
        or candidate.family == "ensemble"
        and isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0
        and isinstance(trees, int) and not isinstance(trees, bool) and trees > 0
        and isinstance(weight, (int, float)) and not isinstance(weight, bool)
        and math.isfinite(float(weight)) and 0.0 <= float(weight) <= 1.0
        or candidate.family == "geometry_gate"
        and isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > 0
        and isinstance(trees, int) and not isinstance(trees, bool) and trees > 0
    )


def _valid_locked_feature_contract(candidate: Candidate) -> bool:
    if candidate.family == "geometry_gate":
        try:
            _validate_geometry_gate_candidate(candidate)
            return True
        except ValueError:
            return False
    valid_pairs = {
        (LEGACY_FEATURES, TRANSFORMATION_VERSION),
        (GEOMETRY_REGIME_FEATURES, GEOMETRY_REGIME_TRANSFORMATION_VERSION),
        (
            LEGACY_GEOMETRY_AUGMENTATION_FEATURES,
            LEGACY_GEOMETRY_AUGMENTATION_TRANSFORMATION_VERSION,
        ),
    }
    if (
        candidate.family not in {"neural_network", "xgboost", "ensemble"}
        or (
            candidate.ordered_prediction_features,
            candidate.feature_transformation_version,
        ) not in valid_pairs
    ):
        return False
    if candidate.family != "ensemble":
        return True
    try:
        neural = _candidate_from_dict(candidate.parameters["neural_network"])
        xgboost = _candidate_from_dict(candidate.parameters["xgboost"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        neural.family == "neural_network"
        and xgboost.family == "xgboost"
        and neural.ordered_prediction_features == candidate.ordered_prediction_features
        and xgboost.ordered_prediction_features == candidate.ordered_prediction_features
        and neural.feature_transformation_version
        == candidate.feature_transformation_version
        and xgboost.feature_transformation_version
        == candidate.feature_transformation_version
        and _valid_locked_feature_contract(neural)
        and _valid_locked_feature_contract(xgboost)
    )


def _valid_locked_contract(contract: Mapping[str, Any]) -> bool:
    evidence = contract.get("development_evidence")
    feature_contract = contract.get("feature_contract")
    fixed_counts = contract.get("fixed_training_counts")
    model_specification = contract.get("model_specification")
    seeds = contract.get("selection_seeds")
    development_sources = contract.get("development_source_groups")
    development_usage = contract.get("development_data_usage")
    required_development_stages = set(DEVELOPMENT_ONLY_STAGES)
    try:
        candidate = _candidate_from_dict(contract["candidate"])
    except (KeyError, TypeError, ValueError):
        return False
    if not _valid_locked_feature_contract(candidate):
        return False
    if contract.get("development_contract") == "explicit_train_validation":
        required_usage = {
            "fitting": "training_records_only",
            "preprocessing": "training_records_only",
            "early_stopping": "validation_records_only",
            "ensemble_selection": "validation_records_only",
            "threshold_selection": "validation_records_only",
            "candidate_selection": "validation_records_only",
            "candidate_locking": "training_and_validation_contract_only",
            **({"gate_fitting": "training_oof_predictions_only"}
               if candidate.family == "geometry_gate" else {}),
        }
        return (
            contract.get("version") == TUNING_VERSION
            and contract.get("output_unit") == "g"
            and contract.get("final_test_access") is False
            and contract.get("test_input_accessed") is False
            and contract.get("refit_partition") == "training_records_only"
            and contract.get("runtime_configuration") == _locked_runtime_configuration(candidate)
            and isinstance(evidence, Mapping)
            and all(isinstance(evidence.get(key), str) and evidence.get(key) for key in (
                "search_plan_id", "training_input_fingerprint",
                "validation_input_fingerprint", "normalized_training_fingerprint",
                "normalized_validation_fingerprint", "partition_identity_fingerprint",
                "search_plan_fingerprint", "test_input_attestation",
            ))
            and evidence.get("test_input_attestation") == "no_test_argument_or_path_available"
            and evidence.get("validation_grouping_contract")
            == EXPLICIT_DEVELOPMENT_EVIDENCE_VERSION
            and isinstance(development_sources, list) and bool(development_sources)
            and development_sources == sorted(set(development_sources))
            and all(isinstance(source, str) and source for source in development_sources)
            and development_usage == required_usage
            and isinstance(feature_contract, Mapping)
            and feature_contract.get("ordered_features")
            == list(candidate.ordered_prediction_features)
            and contract.get("features") == list(candidate.ordered_prediction_features)
            and feature_contract.get(
                "transformation_version",
                TRANSFORMATION_VERSION
                if candidate.ordered_prediction_features == LEGACY_FEATURES else None,
            ) == candidate.feature_transformation_version
            and feature_contract.get("dtype") == "float32"
            and isinstance(fixed_counts, Mapping)
            and _valid_fixed_training_counts(candidate, fixed_counts)
            and model_specification
            == _locked_model_specification(candidate, fixed_counts).to_dict()
            and isinstance(seeds, list) and len(seeds) == 2
            and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
            and seeds[0] != seeds[1]
        )
    return (
        contract.get("version") == TUNING_VERSION
        and contract.get("output_unit") == "g"
        and contract.get("final_test_access") is False
        and contract.get("refit_partition") == "all_included_development_records"
        and contract.get("runtime_configuration") == _locked_runtime_configuration(candidate)
        and isinstance(evidence, Mapping)
        and all(isinstance(evidence.get(key), str) and evidence.get(key) for key in (
            "search_plan_id", "input_fingerprint", "normalized_input_fingerprint",
            "source_allocation_fingerprint", "search_plan_fingerprint",
            "development_split_fingerprint",
        ))
        and isinstance(development_sources, list) and bool(development_sources)
        and development_sources == sorted(set(development_sources))
        and all(isinstance(source, str) and source for source in development_sources)
        and isinstance(development_usage, Mapping)
        and set(development_usage) == required_development_stages
        and all(value == "development_records_only"
                for value in development_usage.values())
        and isinstance(feature_contract, Mapping)
        and feature_contract.get("ordered_features")
        == list(candidate.ordered_prediction_features)
        and contract.get("features") == list(candidate.ordered_prediction_features)
        and feature_contract.get(
            "transformation_version",
            TRANSFORMATION_VERSION
            if candidate.ordered_prediction_features == LEGACY_FEATURES else None,
        ) == candidate.feature_transformation_version
        and feature_contract.get("dtype") == "float32"
        and isinstance(fixed_counts, Mapping)
        and _valid_fixed_training_counts(candidate, fixed_counts)
        and (
            model_specification is None  # Backward-compatible v5 lock.
            or model_specification
            == _locked_model_specification(candidate, fixed_counts).to_dict()
        )
        and isinstance(seeds, list) and len(seeds) == 2
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        and seeds[0] != seeds[1]
    )


def _validate_geometry_gate_candidate(candidate: Candidate) -> None:
    try:
        neural = _candidate_from_dict(candidate.parameters["neural_network"])
        xgboost = _candidate_from_dict(candidate.parameters["xgboost"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid_candidate_configuration") from None
    if (
        candidate.family != "geometry_gate"
        or candidate.ordered_prediction_features
        != CROSS_FITTED_GEOMETRY_GATE_FEATURES
        or candidate.feature_transformation_version
        != CROSS_FITTED_GEOMETRY_GATE_TRANSFORMATION_VERSION
        or set(candidate.parameters) != {
            "component_rank", "selection_partition", "neural_network", "xgboost",
            "cross_fit_folds", "cross_fit_assignment", "gate_fit_partition",
            "gate_features", "gate_feature_transformation_version",
            "ridge_penalty", "weight_constraint",
        }
        or candidate.parameters["component_rank"] not in (1, 2, 3)
        or candidate.parameters["selection_partition"] != "validation_records_only"
        or candidate.parameters["cross_fit_folds"] != CROSS_FIT_FOLDS
        or candidate.parameters["cross_fit_assignment"]
        != "stable_record_identity_hash_without_source_metadata"
        or candidate.parameters["gate_fit_partition"]
        != "training_oof_predictions_only"
        or tuple(candidate.parameters["gate_features"]) != GEOMETRY_REGIME_FEATURES
        or candidate.parameters["gate_feature_transformation_version"]
        != GEOMETRY_REGIME_TRANSFORMATION_VERSION
        or candidate.parameters["ridge_penalty"] not in GEOMETRY_GATE_RIDGE_PENALTIES
        or candidate.parameters["weight_constraint"] != "clip_0_1"
        or neural.family != "neural_network" or xgboost.family != "xgboost"
        or neural.ordered_prediction_features != LEGACY_FEATURES
        or xgboost.ordered_prediction_features != LEGACY_FEATURES
    ):
        raise ValueError("invalid_candidate_configuration")
    _validate_declared_candidate(neural)
    _validate_declared_candidate(xgboost)


def _validate_declared_candidate(candidate: Candidate) -> None:
    # The deep model-definition module owns structural and semantic validation;
    # this additional check limits governed search candidates to the declared domain.
    candidate_model_specification(
        candidate.family, candidate.parameters,
        ordered_prediction_features=candidate.ordered_prediction_features,
    )
    matching_domains = [
        domain
        for policy in SEARCH_PLAN_POLICIES.values()
        for family, domain in _supported_domains(policy).items()
        if family == candidate.family
        and candidate.ordered_prediction_features == policy.prediction_features
        and candidate.feature_transformation_version
        == policy.feature_transformation_version
        and set(candidate.parameters) == set(domain)
        and all(
            _supported_domain_value(value, domain[key])
            for key, value in candidate.parameters.items()
        )
    ]
    if not matching_domains:
        raise ValueError("invalid_candidate_configuration")
    if candidate.family == "xgboost" and (
        candidate.parameters["objective"] != "reg:squarederror"
        or candidate.parameters["n_jobs"] != 1
    ):
        raise ValueError("invalid_candidate_configuration")


def _supported_domain_value(value: Any, supported: Sequence[Any]) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    for expected in supported:
        if isinstance(expected, tuple):
            if (isinstance(value, tuple) and len(value) == len(expected)
                    and all(_supported_domain_value(item, (wanted,))
                            for item, wanted in zip(value, expected))):
                return True
        elif isinstance(expected, bool):
            if isinstance(value, bool) and value == expected:
                return True
        elif isinstance(expected, int):
            if isinstance(value, int) and not isinstance(value, bool) and value == expected:
                return True
        elif isinstance(expected, float):
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value)) and float(value) == expected):
                return True
        elif type(value) is type(expected) and value == expected:
            return True
    return False


def _validate_plan(
    plan: SearchPlan, limits: SearchLimits, input_fingerprint: str,
    dependencies: Mapping[str, str], identities: SearchPlanIdentities,
) -> None:
    _validate_limits(limits)
    expected_plan = _generate_bound_search_plan(
        limits, input_fingerprint, dependencies, identities,
    )
    if plan != expected_plan:
        raise ValueError("invalid_search_plan")
    families = [candidate.family for candidate in plan.component_trials]
    if (
        families.count("neural_network") != limits.neural_network_trials
        or families.count("xgboost") != limits.xgboost_trials
    ):
        raise ValueError("invalid_search_plan")
    if len({candidate.candidate_id for candidate in plan.component_trials}) != (
        limits.neural_network_trials + limits.xgboost_trials
    ):
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
    from ..evaluation.evidence import review_public_summary
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


def _search_stop_reason(result: TuningResult) -> str:
    return result.blockers[0] if result.blockers else "search_stopped"


def _skipped_candidates(result: TuningResult) -> list[dict[str, Any]]:
    attempted = {run.candidate.candidate_id for run in result.initial_results}
    reason = _search_stop_reason(result)
    skipped = [
        {"candidate_id": candidate.candidate_id, "phase": "initial", "reason": reason}
        for candidate in result.plan.component_trials if candidate.candidate_id not in attempted
    ]
    skipped.extend(
        {"candidate_id": f"ensemble-slot-{index + 1:02d}", "phase": "initial",
         "reason": reason}
        for index in range(
            result.allocation["ensemble"], len(result.plan.ensemble_rules)
        )
    )
    eligible_count = sum(run.eligible for run in result.initial_results)
    initial_candidate_count = (
        len(result.plan.component_trials) + len(result.plan.ensemble_rules)
    )
    second_seed_count = int(result.plan.second_seed_rule["candidate_count"])
    skipped.extend(
        {"candidate_id": f"second-seed-slot-{index + 1:02d}", "phase": "second_seed",
         "reason": (
             "eligible_candidate_shortfall"
             if len(result.initial_results) == initial_candidate_count
             and index >= eligible_count else reason
         )}
        for index in range(result.allocation["second_seed"], second_seed_count)
    )
    maximum_runs = int(result.plan.resource_limits["maximum_candidate_runs"])
    return skipped[:max(0, maximum_runs - result.run_count)]


def _second_seed_comparison(result: TuningResult) -> dict[str, Any]:
    eligible_count = sum(
        run.status == "completed" and run.eligible for run in result.initial_results
    )
    target = min(result.plan.second_seed_rule["candidate_count"], eligible_count)
    return {
        "planned_finalists": result.plan.second_seed_rule["candidate_count"],
        "eligible_initial_candidates": eligible_count,
        "repeated_candidates": len(result.second_seed_results),
        "shortfall": max(0, result.plan.second_seed_rule["candidate_count"] - eligible_count),
        "complete": (
            len(result.initial_results)
            == len(result.plan.component_trials) + len(result.plan.ensemble_rules)
            and len(result.second_seed_results) == target
            and all(run.status == "completed" for run in result.second_seed_results)
        ),
    }


def _initial_promotable_ranking(result: TuningResult) -> list[dict[str, Any]]:
    ranked = sorted(
        (run for run in result.initial_results
         if run.status == "completed" and run.eligible),
        key=_rank_key,
    )
    return [
        {
            "rank": rank,
            "candidate_id": run.candidate.candidate_id,
            "family": run.candidate.family,
            "metrics": dict(run.metrics),
            "rationale": list(result.plan.ranking_rule),
        }
        for rank, run in enumerate(ranked, 1)
    ]


def _promotable_ranking(result: TuningResult) -> list[dict[str, Any]]:
    ranked = sorted(
        (item for item in result.combined_results if item["eligible"]),
        key=_combined_rank_key,
    )
    return [
        {
            "rank": rank,
            "candidate_id": item["candidate_id"],
            "family": item["family"],
            "metrics": dict(item["metrics"]),
            "rationale": list(result.plan.ranking_rule),
            "seed_results": list(item["seed_results"]),
            "seed_eligibility": list(item["seed_eligibility"]),
            "equal_seed_weight": item["equal_seed_weight"],
        }
        for rank, item in enumerate(ranked, 1)
    ]


def _best_development_result(result: TuningResult) -> dict[str, Any] | None:
    completed = [run for run in result.initial_results if run.status == "completed"]
    if not completed:
        return None
    best = min(completed, key=_rank_key)
    return {
        "candidate": asdict(best.candidate),
        "seed": best.seed,
        "eligible": best.eligible,
        "metrics": dict(best.metrics),
        "blockers": list(best.blockers),
    }


def _selected_ensemble_weights(runs: Sequence[CandidateRun]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": run.candidate.candidate_id,
            "fold_weights": [
                {
                    "source": report["source"],
                    "neural_network_weight": metadata["selected_neural_network_weight"],
                }
                for report, metadata in zip(run.source_reports, run.fit_metadata)
            ],
        }
        for run in runs if run.candidate.family == "ensemble"
    ]


def _candidate_history(
    result: TuningResult, skipped: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    if result.control_result is not None:
        history.append({
            "candidate_id": result.control_result.candidate.candidate_id,
            "allocation": "control",
            "status": result.control_result.status,
            "reason": result.control_result.blockers[0]
            if result.control_result.blockers else None,
        })
    history.extend({
        "candidate_id": run.candidate.candidate_id,
        "allocation": "initial",
        "status": run.status,
        "reason": run.blockers[0] if run.blockers else None,
    } for run in result.initial_results)
    history.extend({
        "candidate_id": run.candidate.candidate_id,
        "allocation": "second_seed",
        "status": run.status,
        "reason": run.blockers[0] if run.blockers else None,
    } for run in result.second_seed_results)
    history.extend({
        "candidate_id": item["candidate_id"],
        "allocation": item["phase"],
        "status": "skipped",
        "reason": item["reason"],
    } for item in skipped)
    return history


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
    """Candidate-workflow adapter over the explicit model-definition module."""

    def __init__(self) -> None:
        self.models = ModelRuntime(TensorflowXGBoostBackend())
        self.dependency_versions = self.models.dependency_versions

    def fit_fold(self, candidate, seed, train_features, train_targets,
                 validation_features, validation_targets, *, prediction_ranker=None):
        training = TrainingData(tuple(train_features), tuple(train_targets))
        validation = ValidationData(
            tuple(validation_features), tuple(validation_targets), prediction_ranker,
        )
        if candidate.family == "control":
            fitted = self.models.fit(
                fixed_model_specification(), training, validation, seed=seed
            )
            return _candidate_fold_fit(fitted)
        specification = candidate.specification
        if specification is None:
            raise ValueError("invalid candidate specification")
        fitted = self.models.fit(specification, training, validation, seed=seed)
        return _candidate_fold_fit(fitted)

    def refit(self, candidate, seed, features, targets, fixed_training_counts):
        training = TrainingData(tuple(features), tuple(targets))
        specification = _locked_model_specification(candidate, fixed_training_counts)
        fitted = self.models.fit(specification, training, None, seed=seed)
        return LockedFit(
            fitted.predictor, dict(fitted.preprocessing_state), self.models.save(fitted),
            {"seed": seed, **dict(fitted.metadata)},
        )

    def serialize_fold(self, fitted):
        model = fitted.fitted_state.get("_fitted_model")
        if not isinstance(model, FittedModel):
            raise ValueError("model artifact unavailable")
        return self.models.save(model)

    def load_locked(self, candidate, directory, contract):
        specification = _locked_model_specification(
            candidate, contract["fixed_training_counts"]
        )
        preprocessing = json.loads((directory / "preprocessing-state.json").read_text())
        if candidate.family == "geometry_gate":
            assert isinstance(specification, GeometryGateModelSpecification)
            gate_state = json.loads((directory / "gate-state.json").read_text())
            if not _valid_geometry_gate_state(gate_state) or preprocessing.get("gate") != gate_state:
                raise ValueError("invalid_geometry_gate_state")
            neural = self.models.load(
                specification.neural_network,
                {"model.keras": (directory / "neural-model.keras").read_bytes()},
                preprocessing["neural_network"],
            )
            xgboost = self.models.load(
                specification.xgboost,
                {"model.json": (directory / "xgboost-model.json").read_bytes()},
                preprocessing["xgboost"],
            )

            return _geometry_gate_predictor(gate_state, neural, xgboost)
        assert isinstance(specification, ModelSpecification)
        artifact_names = (
            ("model.keras",) if specification.model_kind is ModelKind.NEURAL_NETWORK
            else ("model.json",) if specification.model_kind is ModelKind.XGBOOST
            else ("neural-model.keras", "xgboost-model.json")
        )
        artifacts = {name: (directory / name).read_bytes() for name in artifact_names}
        return self.models.load(specification, artifacts, preprocessing)


def _candidate_fold_fit(fitted: FittedModel) -> CandidateFoldFit:
    if fitted.specification.model_kind is ModelKind.ENSEMBLE:
        ensemble = fitted.specification.ensemble
        assert ensemble is not None
        neural = fitted.backend_state["neural_network"]
        xgboost = fitted.backend_state["xgboost"]
        assert isinstance(neural, FittedModel) and isinstance(xgboost, FittedModel)
        state = {
            "neural_network": _component_fitted_state(neural),
            "xgboost": _component_fitted_state(xgboost),
            "weight": ensemble.neural_network_weight,
            "_fitted_model": fitted,
        }
        metadata = {
            "neural_network": _component_fit_metadata(neural),
            "xgboost": _component_fit_metadata(xgboost),
            "selected_neural_network_weight": ensemble.neural_network_weight,
        }
    else:
        state = {**_component_fitted_state(fitted), "_fitted_model": fitted}
        metadata = _component_fit_metadata(fitted)
    return CandidateFoldFit(fitted.predictor, metadata, state)


def _component_fitted_state(fitted: FittedModel) -> dict[str, Any]:
    state = dict(fitted.preprocessing_state)
    fingerprint_value = fitted.metadata.get("fitted_parameter_fingerprint")
    if fingerprint_value is not None:
        state["fitted_parameter_fingerprint"] = fingerprint_value
    return state


def _component_fit_metadata(fitted: FittedModel) -> dict[str, Any]:
    return {key: value for key, value in fitted.metadata.items()
            if key in {"selected_epochs", "selected_trees"}}


def _locked_model_specification(
    candidate: Candidate, fixed_training_counts: Mapping[str, int | float],
):
    if candidate.family == "geometry_gate":
        _validate_geometry_gate_candidate(candidate)
        fixed_neural, fixed_xgboost = _fixed_geometry_gate_components(
            candidate, fixed_training_counts
        )
        neural_specification = fixed_neural.specification
        xgboost_specification = fixed_xgboost.specification
        if neural_specification is None or xgboost_specification is None:
            raise ValueError("invalid_candidate_configuration")
        return GeometryGateModelSpecification(
            neural_specification,
            xgboost_specification,
            candidate.ordered_prediction_features,
            tuple(candidate.parameters["gate_features"]),
            int(candidate.parameters["cross_fit_folds"]),
            float(candidate.parameters["ridge_penalty"]),
        )
    if candidate.family == "ensemble":
        neural_candidate = _candidate_from_dict(candidate.parameters["neural_network"])
        xgboost_candidate = _candidate_from_dict(candidate.parameters["xgboost"])
        neural_parameters = dict(neural_candidate.parameters)
        neural_parameters["maximum_epochs"] = fixed_training_counts["neural_network_epochs"]
        xgboost_parameters = dict(xgboost_candidate.parameters)
        xgboost_parameters["n_estimators"] = fixed_training_counts["xgboost_trees"]
        return ensemble_model_specification(
            candidate_model_specification(
                "neural_network", neural_parameters,
                ordered_prediction_features=neural_candidate.ordered_prediction_features,
                identity_namespace=(
                    f"minires-model-definition-v1:{neural_candidate.feature_transformation_version}"
                    if neural_candidate.ordered_prediction_features != LEGACY_FEATURES
                    else "minires-model-definition-v1"
                ),
            ),
            candidate_model_specification(
                "xgboost", xgboost_parameters,
                ordered_prediction_features=xgboost_candidate.ordered_prediction_features,
                identity_namespace=(
                    f"minires-model-definition-v1:{xgboost_candidate.feature_transformation_version}"
                    if xgboost_candidate.ordered_prediction_features != LEGACY_FEATURES
                    else "minires-model-definition-v1"
                ),
            ),
            float(fixed_training_counts["ensemble_neural_network_weight"]),
        )
    parameters = dict(candidate.parameters)
    if candidate.family == "neural_network":
        parameters["maximum_epochs"] = fixed_training_counts["neural_network_epochs"]
    elif candidate.family == "xgboost":
        parameters["n_estimators"] = fixed_training_counts["xgboost_trees"]
    return candidate_model_specification(
        candidate.family, parameters,
        ordered_prediction_features=candidate.ordered_prediction_features,
        identity_namespace=(
            f"minires-model-definition-v1:{candidate.feature_transformation_version}"
            if candidate.ordered_prediction_features != LEGACY_FEATURES
            else "minires-model-definition-v1"
        ),
    )


def _selected_epoch_count(
    validation_mae: Sequence[float], minimum_improvement: float,
) -> int:
    if not validation_mae:
        raise ValueError("validation history unavailable")
    best = math.inf
    selected = 1
    for epoch, observed in enumerate(validation_mae, 1):
        value = float(observed)
        if math.isfinite(value) and value < best - minimum_improvement:
            best = value
            selected = epoch
    return selected


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

    def serialize_fold(self, *args: Any, **kwargs: Any) -> Mapping[str, bytes]:
        raise RuntimeError("candidate runtime unavailable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run private bounded MiniRes candidate development."
    )
    parser.add_argument("--training-records", required=True, type=Path)
    parser.add_argument("--validation-records", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--volume-unit", default="mm3")
    parser.add_argument("--scope-confirmed", action="store_true")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--plan-kind", choices=tuple(SEARCH_PLAN_POLICIES),
        default="baseline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        governed_predeclared_plans = {
            "cross_fitted_geometry_gate", "legacy_geometry_augmentation",
            "tail_aligned_selection",
        }
        if args.plan_kind in governed_predeclared_plans:
            _verify_development_artifact_manifest(
                args.training_records, args.validation_records
            )
        try:
            runtime: CandidateRuntime = TensorflowXGBoostCandidateRuntime()
        except ImportError:
            runtime = _BlockedCandidateRuntime("candidate_tuning_dependencies_required")
        except RuntimeError:
            runtime = _BlockedCandidateRuntime("candidate_tuning_runtime_unavailable")
        if args.plan_kind in governed_predeclared_plans:
            _verify_predeclared_environment(runtime.dependency_versions)
        limits = SearchLimits.for_plan(args.seed, args.plan_kind)
        result = develop_candidates(
            args.training_records, args.validation_records,
            EvaluationConfig(None, args.volume_unit,
                             True if args.scope_confirmed else None, seed=args.seed),
            runtime=runtime, output_root=args.output_root, limits=limits,
        )
    except InputError as error:
        raise SystemExit(str(error)) from None
    except (OSError, ValueError, TypeError):
        raise SystemExit("candidate_tuning_failed") from None
    print(json.dumps(result.to_dict(public=True), sort_keys=True, allow_nan=False))
    return 0


def _verify_development_artifact_manifest(
    training_records: Path, validation_records: Path,
) -> None:
    if training_records.parent.resolve() != validation_records.parent.resolve():
        raise InputError("development_artifact_checksum_mismatch")
    try:
        manifest = json.loads((training_records.parent / "manifest.json").read_text())
        artifacts = manifest["artifacts"]
        for path in (training_records, validation_records):
            expected = artifacts[path.name]
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("checksum mismatch")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        raise InputError("development_artifact_checksum_mismatch") from None


def _verify_predeclared_environment(
    dependency_versions: Mapping[str, str],
) -> None:
    try:
        scikit_learn = package_version("scikit-learn")
    except PackageNotFoundError:
        raise InputError("candidate_tuning_dependency_mismatch") from None
    if (
        platform.python_version_tuple()[:2] != ("3", "13")
        or dict(dependency_versions) != CROSS_FITTED_GATE_DEPENDENCY_VERSIONS
        or scikit_learn != CROSS_FITTED_GATE_SCIKIT_LEARN_VERSION
    ):
        raise InputError("candidate_tuning_dependency_mismatch")


def _validate_limits(limits: SearchLimits) -> None:
    values = asdict(limits)
    if not isinstance(limits.seed, int) or isinstance(limits.seed, bool):
        raise ValueError("invalid_search_plan")
    if limits.second_seed is not None and (
        not isinstance(limits.second_seed, int) or isinstance(limits.second_seed, bool)
        or limits.second_seed == limits.seed
    ):
        raise ValueError("invalid_search_plan")
    policy = _plan_policy(limits.plan_kind)
    if policy.fixed_seeds is not None and (
        limits.seed, limits.resolved_second_seed
    ) != policy.fixed_seeds:
        raise ValueError("invalid_search_plan")
    expected = (
        policy.neural_network_trials, policy.xgboost_trials, policy.ensemble_trials,
        policy.second_seed_candidates, policy.maximum_candidate_runs,
    )
    actual = (limits.neural_network_trials, limits.xgboost_trials, limits.ensemble_trials,
              limits.second_seed_candidates, limits.maximum_candidate_runs)
    if actual != expected:
        raise ValueError("invalid_search_plan")
    if not isinstance(limits.maximum_elapsed_seconds, (int, float)) or isinstance(
        limits.maximum_elapsed_seconds, bool
    ) or not math.isfinite(limits.maximum_elapsed_seconds) or not (
        0 < limits.maximum_elapsed_seconds <= policy.maximum_elapsed_seconds
    ):
        raise ValueError("invalid_search_plan")
    if not limits.ensemble_neural_network_weights or any(
        not isinstance(weight, (int, float)) or isinstance(weight, bool)
        or not math.isfinite(weight) or not 0 <= weight <= 1
        for weight in limits.ensemble_neural_network_weights
    ):
        raise ValueError("invalid_search_plan")
    if tuple(sorted(set(limits.ensemble_neural_network_weights))) != limits.ensemble_neural_network_weights:
        raise ValueError("invalid_search_plan")
    if any(isinstance(value, float) and not math.isfinite(value)
           for value in values.values()):
        raise ValueError("invalid_search_plan")


if __name__ == "__main__":
    raise SystemExit(main())
