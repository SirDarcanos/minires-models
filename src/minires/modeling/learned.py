"""Fixed-configuration learned baselines fitted inside frozen evaluation folds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import platform
import time
from typing import Any, Callable, Protocol, Sequence

from ..ingestion import CanonicalRow, fingerprint
from .legacy import LEGACY_FEATURES, prepare_canonical_legacy_features
from .definitions import (
    FittedModel, ModelRuntime, TensorflowXGBoostBackend, TrainingData,
    ValidationData, combine_ensemble_predictions, ensemble_with_weight,
    fixed_model_specification,
)


BatchPredictor = Callable[[Sequence[tuple[float, ...]]], Sequence[float]]
LEARNED_BASELINE_VERSION = "minires-clean-fixed-baselines-v1"


def enable_synchronous_dataset_execution() -> bool:
    """Configure TensorFlow before any model runtime can create a dataset."""
    try:
        import tensorflow as tf
    except ImportError:
        return False
    try:
        tf.data.experimental.enable_debug_mode()
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class LearnedBaselineConfig:
    """The bounded, verified legacy configurations; none are search ranges."""

    seed: int = 34
    batch_size: int = 256
    neural_network_max_epochs: int = 100
    neural_network_early_stopping_patience: int = 8
    neural_network_early_stopping_min_delta: float = 0.005
    neural_network_learning_rate: float = 0.0006350310563507329
    neural_network_l2: float = 3.436477039390904e-06
    neural_network_lr_reduction_factor: float = 0.5
    neural_network_lr_reduction_patience: int = 3
    neural_network_min_learning_rate: float = 1e-8
    xgboost_estimators: int = 900
    xgboost_early_stopping_rounds: int = 50
    xgboost_max_depth: int = 9
    xgboost_learning_rate: float = 0.01
    xgboost_subsample: float = 0.7
    xgboost_colsample_bytree: float = 0.9
    xgboost_n_jobs: int = 1
    ensemble_neural_network_weights: tuple[float, ...] = (0.2,)
    maximum_folds: int = 32
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-6


class FoldRuntime(Protocol):
    dependency_versions: dict[str, str]

    def fit_fold(
        self,
        train_features: Sequence[tuple[float, ...]],
        train_targets: Sequence[float],
        validation_features: Sequence[tuple[float, ...]],
        validation_targets: Sequence[float],
        config: LearnedBaselineConfig,
        artifact_dir: Path | None,
    ) -> "FittedFold": ...


@dataclass(frozen=True)
class FittedFold:
    neural_network: BatchPredictor
    xgboost: BatchPredictor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LearnedBaseline:
    """Refit the fixed neural-network, XGBoost, and bounded ensemble baselines."""

    config: LearnedBaselineConfig = field(default_factory=LearnedBaselineConfig)
    runtime: FoldRuntime | None = field(default=None, repr=False, compare=False)
    name: str = "clean_fixed_ensemble"

    @property
    def contract(self) -> dict[str, Any]:
        config = asdict(self.config)
        specification = fixed_model_specification()
        assert specification.ensemble is not None
        neural = specification.ensemble.neural_network
        xgboost = specification.ensemble.xgboost
        return {
            "classification": "clean_fixed_configuration_baseline",
            "version": LEARNED_BASELINE_VERSION,
            "features": list(LEGACY_FEATURES),
            "feature_units": ["unit_unknown", "mm3", "mm2", "mm3", "dimensionless", "unit_unknown", "mm-1"],
            "preprocessing": {
                "neural_network": "float32; per-feature normalization fitted on fold train only",
                "xgboost": "same float32 matrix without normalization",
                "normalization_fit_partition": "fold_train_only",
                "dataset_execution": "synchronous deterministic tf.data execution",
                "legacy_versus_new": (
                    "no whole-dataset tail filter, integer cast, rounding, ratio recomputation, "
                    "or embedded released normalization; clean NN normalization is fitted per fold"
                ),
            },
            "model_specification": specification.to_dict(),
            "neural_network": {
                "layers": [
                    {"units": units, "activation": activation, "dropout": dropout}
                    for units, activation, dropout in neural.architecture_parameters["layer_specs"]
                ] + [{"units": 1, "activation": "linear", "dropout": 0.0}],
                "optimizer": neural.architecture_parameters["optimizer"],
                "loss": neural.architecture_parameters["loss"],
                "monitor": "val_mean_absolute_error",
                "configuration": {key: value for key, value in config.items() if key.startswith("neural_network_") or key == "batch_size"},
            },
            "xgboost": {
                "objective": xgboost.architecture_parameters["objective"],
                "tree_method": "hist", "eval_metric": "mae",
                "configuration": {key: value for key, value in config.items() if key.startswith("xgboost_")},
            },
            "ensemble": {
                "selection_metric": "validation_mae_g",
                "selection_partition": "common_inner_validation_only",
                "candidate_neural_network_weights": list(self.config.ensemble_neural_network_weights),
            },
            "run_configuration": config,
            "numerical_reproducibility": {
                "absolute_tolerance": self.config.absolute_tolerance,
                "relative_tolerance": self.config.relative_tolerance,
                "determinism": "same manifest, dependencies, platform, and seed",
            },
            "output_unit": "g",
        }


@dataclass(frozen=True)
class FoldOutput:
    source: str
    test_rows: tuple[int, ...]
    actual: tuple[float, ...]
    volume_mm3: tuple[float, ...]
    neural_network: tuple[float, ...]
    xgboost: tuple[float, ...]
    ensemble: tuple[float, ...]
    neural_network_weight: float
    fit_audit: dict[str, Any]
    fit_metadata: dict[str, Any]


@dataclass(frozen=True)
class LearnedRun:
    status: str
    blockers: tuple[str, ...]
    folds: tuple[FoldOutput, ...]
    contract: dict[str, Any]
    dependency_versions: dict[str, str]


def fit_frozen_folds(
    baseline: LearnedBaseline,
    rows: Sequence[CanonicalRow],
    manifest: dict[str, Any],
    artifact_root: Path | None,
) -> LearnedRun:
    """Fit and score each frozen fold without exposing holdouts to the runtime."""
    started = time.monotonic()
    contract = baseline.contract
    if manifest["status"] != "frozen_source_holdout":
        return LearnedRun("blocked", tuple(manifest["blockers"]), (), contract, {})
    if len(manifest["folds"]) > baseline.config.maximum_folds:
        return LearnedRun("blocked", ("learned_baseline_fold_limit_exceeded",), (), contract, {})
    if not _valid_config(baseline.config):
        return LearnedRun("blocked", ("invalid_learned_baseline_configuration",), (), contract, {})
    runtime: FoldRuntime
    if baseline.runtime is None:
        try:
            runtime = TensorflowXGBoostRuntime()
        except (ImportError, RuntimeError):
            return LearnedRun("blocked", ("learned_baseline_dependencies_required",), (), contract, {})
    else:
        runtime = baseline.runtime
    by_index = {row.row_index: row for row in rows if row.outcome == "included"}
    outputs: list[FoldOutput] = []
    try:
        for fold_number, fold in enumerate(manifest["folds"]):
            train = [by_index[index] for index in fold["train"]]
            validation = [by_index[index] for index in fold["validation"]]
            test = [by_index[index] for index in fold["test"]]
            train_x, train_y = _matrix(train)
            validation_x, validation_y = _matrix(validation)
            test_x, test_y = _matrix(test)
            fold_dir = artifact_root / f"fold-{fold_number:03d}" if artifact_root else None
            fitted = runtime.fit_fold(train_x, train_y, validation_x, validation_y,
                                      baseline.config, fold_dir)
            validation_nn = _predict(fitted.neural_network, validation_x)
            validation_xgb = _predict(fitted.xgboost, validation_x)
            weight = _select_weight(validation_y, validation_nn, validation_xgb,
                                    baseline.config.ensemble_neural_network_weights)
            test_nn = _predict(fitted.neural_network, test_x)
            test_xgb = _predict(fitted.xgboost, test_x)
            ensemble = combine_ensemble_predictions(
                ensemble_with_weight(fixed_model_specification(), weight),
                test_nn,
                test_xgb,
            )
            state = {"train_features": train_x, "train_targets": train_y,
                     "validation_features": validation_x, "validation_targets": validation_y,
                     "configuration": asdict(baseline.config), "fit_metadata": fitted.metadata,
                     "selected_weight": weight}
            outputs.append(FoldOutput(
                source=fold["source"], test_rows=tuple(fold["test"]), actual=test_y,
                volume_mm3=tuple(float(row.features["volume_mm3"]) for row in test
                                 if row.features["volume_mm3"] is not None),
                neural_network=test_nn, xgboost=test_xgb, ensemble=ensemble,
                neural_network_weight=weight,
                fit_audit={
                    "train_rows": list(fold["train"]),
                    "validation_rows": list(fold["validation"]),
                    "test_rows": list(fold["test"]),
                    "ensemble_selection_rows": list(fold["validation"]),
                    "fitted_state_fingerprint": fingerprint(state),
                    "test_data_fingerprint": fingerprint({"features": test_x, "targets": test_y}),
                },
                fit_metadata=fitted.metadata,
            ))
    except Exception:
        # Third-party failures can contain paths or private values; return a bounded code.
        return LearnedRun("blocked", ("learned_baseline_fit_failed",), (), contract,
                          dict(getattr(runtime, "dependency_versions", {})))
    contract["run"] = {
        "runtime_seconds": time.monotonic() - started,
        "fold_count": len(outputs),
        "split_fingerprint": fingerprint(manifest),
        "dependency_versions": dict(runtime.dependency_versions),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    return LearnedRun("completed", (), tuple(outputs), contract,
                      dict(runtime.dependency_versions))


def _matrix(rows: Sequence[CanonicalRow]) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    features = tuple(prepare_canonical_legacy_features(row) for row in rows)
    targets = tuple(float(row.sliced_resin_mass_g) for row in rows if row.sliced_resin_mass_g is not None)
    if len(features) != len(targets):
        raise ValueError("target unavailable")
    return features, targets


def _predict(predictor: BatchPredictor, rows: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    values = tuple(float(value) for value in predictor(rows))
    if len(values) != len(rows) or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid prediction")
    return values


def _select_weight(actual: Sequence[float], neural: Sequence[float], xgboost: Sequence[float],
                   candidates: Sequence[float]) -> float:
    return min(candidates, key=lambda weight: (
        math.fsum(abs(weight * nn + (1.0 - weight) * xgb - target)
                  for nn, xgb, target in zip(neural, xgboost, actual)) / len(actual),
        weight,
    ))


def _valid_config(config: LearnedBaselineConfig) -> bool:
    # Issue #6 is an evaluation of one verified configuration, not a tuning
    # interface. Reject every alteration, including wider candidate sets or
    # unbounded worker counts.
    return config == LearnedBaselineConfig()


class TensorflowXGBoostRuntime:
    """Fixed-baseline adapter over the explicit model-definition module."""

    def __init__(self) -> None:
        self.models = ModelRuntime(TensorflowXGBoostBackend())
        self.dependency_versions = self.models.dependency_versions

    def fit_fold(self, train_features, train_targets, validation_features, validation_targets,
                 config, artifact_dir):
        if config != LearnedBaselineConfig():
            raise ValueError("invalid_learned_baseline_configuration")
        fitted = self.models.fit(
            fixed_model_specification(),
            TrainingData(tuple(train_features), tuple(train_targets)),
            ValidationData(tuple(validation_features), tuple(validation_targets)),
            seed=config.seed,
        )
        neural = fitted.backend_state.get("neural_network")
        xgboost = fitted.backend_state.get("xgboost")
        if not isinstance(neural, FittedModel) or not isinstance(xgboost, FittedModel):
            raise ValueError("fixed_model_fit_failed")
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            artifacts = self.models.save(fitted)
            names = {
                "neural-model.keras": "neural-network.keras",
                "xgboost-model.json": "xgboost.json",
            }
            for name, content in artifacts.items():
                path = artifact_dir / names[name]
                path.write_bytes(content)
                path.chmod(0o600)
        return FittedFold(
            neural_network=neural.predictor,
            xgboost=xgboost.predictor,
            metadata={
                "normalization_mean": neural.preprocessing_state["mean"],
                "normalization_variance": neural.preprocessing_state["variance"],
                "neural_network_epochs": neural.metadata["selected_epochs"],
                "xgboost_best_iteration": int(xgboost.metadata["selected_trees"]) - 1,
                "fitted_parameter_fingerprint": fingerprint({
                    "neural_network": neural.metadata["fitted_parameter_fingerprint"],
                    "xgboost": xgboost.metadata["fitted_parameter_fingerprint"],
                }),
            },
        )
