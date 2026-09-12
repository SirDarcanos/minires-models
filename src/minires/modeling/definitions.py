"""Explicit MiniRes model specifications and their fitting/loading interface.

This is the single entry point for architecture knowledge. Workflow modules choose
specifications and supply data; this module validates, constructs, composes, fits,
serializes, and loads predictors through an injected backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from .legacy import LEGACY_FEATURES


BatchPredictor = Callable[[Sequence[tuple[float, ...]]], Sequence[float]]
MODEL_DEFINITION_VERSION = "minires-model-definition-v1"


class ModelKind(str, Enum):
    """Predictor architecture kind, distinct from a miniature family."""

    NEURAL_NETWORK = "neural_network"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class PreprocessingContract:
    feature_dtype: str
    normalization: str


@dataclass(frozen=True)
class EnsembleDefinition:
    neural_network: "ModelSpecification"
    xgboost: "ModelSpecification"
    neural_network_weight: float


@dataclass(frozen=True)
class ModelSpecification:
    """Complete, validated, content-addressed definition of one predictor."""

    model_kind: ModelKind
    preprocessing: PreprocessingContract
    architecture_parameters: Mapping[str, Any]
    training_parameters: Mapping[str, Any]
    ordered_prediction_features: tuple[str, ...] = LEGACY_FEATURES
    output_unit: str = "g"
    ensemble: EnsembleDefinition | None = None
    identity_namespace: str = MODEL_DEFINITION_VERSION

    def __post_init__(self) -> None:
        architecture = _immutable_mapping(self.architecture_parameters)
        training = _immutable_mapping(self.training_parameters)
        object.__setattr__(self, "architecture_parameters", architecture)
        object.__setattr__(self, "training_parameters", training)
        self.validate()

    def validate(self) -> None:
        if (
            not self.ordered_prediction_features
            or len(set(self.ordered_prediction_features)) != len(self.ordered_prediction_features)
            or self.output_unit != "g"
            or not self.identity_namespace
            or self.preprocessing.feature_dtype != "float32"
        ):
            raise ValueError("invalid_model_specification")
        if self.model_kind is ModelKind.ENSEMBLE:
            if self.ensemble is None or self.architecture_parameters or self.training_parameters:
                raise ValueError("invalid_model_specification")
            if (
                self.ensemble.neural_network.model_kind is not ModelKind.NEURAL_NETWORK
                or self.ensemble.xgboost.model_kind is not ModelKind.XGBOOST
                or self.ensemble.neural_network.ordered_prediction_features
                != self.ordered_prediction_features
                or self.ensemble.xgboost.ordered_prediction_features
                != self.ordered_prediction_features
                or self.ensemble.neural_network.output_unit != self.output_unit
                or self.ensemble.xgboost.output_unit != self.output_unit
                or not _finite_number(self.ensemble.neural_network_weight)
                or not 0.0 <= float(self.ensemble.neural_network_weight) <= 1.0
            ):
                raise ValueError("invalid_model_specification")
        elif self.ensemble is not None:
            raise ValueError("invalid_model_specification")
        elif self.model_kind is ModelKind.NEURAL_NETWORK:
            _validate_neural_parameters(specification_parameters(self))
        elif self.model_kind is ModelKind.XGBOOST:
            _validate_xgboost(specification_parameters(self))

    @property
    def stable_identity(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_identity=False), sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode()
        return f"model-{self.model_kind.value}-{sha256(encoded).hexdigest()[:16]}"

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": self.identity_namespace,
            "model_kind": self.model_kind.value,
            "ordered_prediction_features": list(self.ordered_prediction_features),
            "preprocessing": {
                "feature_dtype": self.preprocessing.feature_dtype,
                "normalization": self.preprocessing.normalization,
            },
            "architecture_parameters": _jsonable(self.architecture_parameters),
            "training_parameters": _jsonable(self.training_parameters),
            "output_unit": self.output_unit,
        }
        if self.ensemble is not None:
            result["ensemble"] = {
                "neural_network": self.ensemble.neural_network.to_dict(),
                "xgboost": self.ensemble.xgboost.to_dict(),
                "neural_network_weight": self.ensemble.neural_network_weight,
            }
        if include_identity:
            result["stable_identity"] = self.stable_identity
        return result

    def describe(self) -> str:
        if self.model_kind is ModelKind.NEURAL_NETWORK:
            layers = [*self.architecture_parameters["layers"], 1]
            shape = " → ".join(str(value) for value in layers)
            return (
                f"neural_network ({shape}; "
                f"{self.architecture_parameters['activation']}; {self.output_unit})"
            )
        if self.model_kind is ModelKind.XGBOOST:
            return (
                "xgboost ("
                f"{self.architecture_parameters['n_estimators']} trees; "
                f"depth {self.architecture_parameters['max_depth']}; {self.output_unit})"
            )
        assert self.ensemble is not None
        return (
            f"ensemble ({self.ensemble.neural_network_weight:g} neural_network + "
            f"{1.0 - self.ensemble.neural_network_weight:g} xgboost; {self.output_unit})"
        )


@dataclass(frozen=True)
class TrainingData:
    features: tuple[tuple[float, ...], ...]
    targets: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_data(self.features, self.targets)


@dataclass(frozen=True)
class ValidationData:
    features: tuple[tuple[float, ...], ...]
    targets: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_data(self.features, self.targets)


@dataclass(frozen=True)
class FittedModel:
    specification: ModelSpecification
    predictor: BatchPredictor
    preprocessing_state: Mapping[str, Any]
    metadata: Mapping[str, Any]
    backend_state: Mapping[str, Any] = field(repr=False, compare=False)


class ModelBackend(Protocol):
    @property
    def dependency_versions(self) -> Mapping[str, str]: ...

    def fit_component(
        self, specification: ModelSpecification, training: TrainingData,
        validation: ValidationData | None, seed: int,
    ) -> FittedModel: ...

    def serialize_component(self, fitted: FittedModel) -> Mapping[str, bytes]: ...

    def load_component(
        self, specification: ModelSpecification, artifacts: Mapping[str, bytes],
        preprocessing_state: Mapping[str, Any],
    ) -> BatchPredictor: ...


class ModelRuntime:
    """Fit, save, and load every model kind through one small interface."""

    def __init__(self, backend: ModelBackend) -> None:
        self.backend = backend
        self.dependency_versions = dict(backend.dependency_versions)

    def fit(
        self, specification: ModelSpecification, training: TrainingData,
        validation: ValidationData | None, *, seed: int,
    ) -> FittedModel:
        specification.validate()
        _validate_feature_width(
            training.features, len(specification.ordered_prediction_features)
        )
        if validation is not None:
            _validate_feature_width(
                validation.features, len(specification.ordered_prediction_features)
            )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("invalid_model_seed")
        if specification.model_kind is not ModelKind.ENSEMBLE:
            return self.backend.fit_component(specification, training, validation, seed)
        assert specification.ensemble is not None
        neural = self.backend.fit_component(
            specification.ensemble.neural_network, training, validation, seed
        )
        xgboost = self.backend.fit_component(
            specification.ensemble.xgboost, training, validation, seed
        )
        weight = float(specification.ensemble.neural_network_weight)
        return FittedModel(
            specification=specification,
            predictor=_ensemble_predictor(
                specification, neural.predictor, xgboost.predictor
            ),
            preprocessing_state={
                "neural_network": dict(neural.preprocessing_state),
                "xgboost": dict(xgboost.preprocessing_state),
            },
            metadata={
                "neural_network": dict(neural.metadata),
                "xgboost": dict(xgboost.metadata),
                "neural_network_weight": weight,
            },
            backend_state={"neural_network": neural, "xgboost": xgboost},
        )

    def save(self, fitted: FittedModel) -> dict[str, bytes]:
        if fitted.specification.model_kind is not ModelKind.ENSEMBLE:
            return _validated_artifacts(self.backend.serialize_component(fitted))
        neural = fitted.backend_state.get("neural_network")
        xgboost = fitted.backend_state.get("xgboost")
        if not isinstance(neural, FittedModel) or not isinstance(xgboost, FittedModel):
            raise ValueError("model_artifact_unavailable")
        return {
            **{f"neural-{name}": content for name, content in
               _validated_artifacts(self.backend.serialize_component(neural)).items()},
            **{f"xgboost-{name}": content for name, content in
               _validated_artifacts(self.backend.serialize_component(xgboost)).items()},
        }

    def load(
        self, specification: ModelSpecification, artifacts: Mapping[str, bytes],
        preprocessing_state: Mapping[str, Any],
    ) -> BatchPredictor:
        specification.validate()
        checked = _validated_artifacts(artifacts)
        if specification.model_kind is not ModelKind.ENSEMBLE:
            return self.backend.load_component(specification, checked, preprocessing_state)
        assert specification.ensemble is not None
        neural_artifacts = _strip_prefix(checked, "neural-")
        xgboost_artifacts = _strip_prefix(checked, "xgboost-")
        neural_state = preprocessing_state.get("neural_network", {})
        xgboost_state = preprocessing_state.get("xgboost", {})
        # v5 locks stored the XGBoost no-op preprocessing marker as a scalar.
        if xgboost_state == "unnormalized_float32":
            xgboost_state = {"xgboost": xgboost_state}
        if not isinstance(neural_state, Mapping) or not isinstance(xgboost_state, Mapping):
            raise ValueError("invalid_preprocessing_state")
        neural = self.backend.load_component(
            specification.ensemble.neural_network, neural_artifacts, neural_state
        )
        xgboost = self.backend.load_component(
            specification.ensemble.xgboost, xgboost_artifacts, xgboost_state
        )
        return _ensemble_predictor(specification, neural, xgboost)


def candidate_model_specification(
    model_kind: str | ModelKind, parameters: Mapping[str, Any]
) -> ModelSpecification:
    """Convert one search candidate into the explicit model-definition interface."""
    try:
        kind = ModelKind(model_kind)
    except (TypeError, ValueError):
        raise ValueError("invalid_model_specification") from None
    copied = _plain_mapping(parameters)
    if kind is ModelKind.NEURAL_NETWORK:
        _validate_neural_candidate(copied)
        architecture_keys = {"layers", "activation", "dropout", "optimizer", "loss", "l2"}
        return ModelSpecification(
            kind,
            PreprocessingContract("float32", "fit_on_training_records"),
            {key: copied[key] for key in architecture_keys},
            {key: value for key, value in copied.items() if key not in architecture_keys},
        )
    if kind is ModelKind.XGBOOST:
        _validate_xgboost(copied)
        training_keys = {"early_stopping_rounds"}
        return ModelSpecification(
            kind,
            PreprocessingContract("float32", "none"),
            {key: value for key, value in copied.items() if key not in training_keys},
            {key: copied[key] for key in training_keys},
        )
    raise ValueError("invalid_model_specification")


def combine_ensemble_predictions(
    specification: ModelSpecification,
    neural_network: Sequence[float],
    xgboost: Sequence[float],
) -> tuple[float, ...]:
    """Compose component predictions only as declared by an ensemble specification."""
    specification.validate()
    if specification.model_kind is not ModelKind.ENSEMBLE or specification.ensemble is None:
        raise ValueError("invalid_model_specification")
    if len(neural_network) != len(xgboost):
        raise ValueError("invalid_model_prediction")
    weight = float(specification.ensemble.neural_network_weight)
    values = tuple(
        weight * float(neural) + (1.0 - weight) * float(tree)
        for neural, tree in zip(neural_network, xgboost)
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("invalid_model_prediction")
    return values


def ensemble_model_specification(
    neural_network: ModelSpecification, xgboost: ModelSpecification,
    neural_network_weight: float,
) -> ModelSpecification:
    return ModelSpecification(
        ModelKind.ENSEMBLE,
        PreprocessingContract("float32", "defined_by_members"),
        {}, {},
        ensemble=EnsembleDefinition(neural_network, xgboost, neural_network_weight),
    )


def ensemble_with_weight(
    specification: ModelSpecification, neural_network_weight: float,
) -> ModelSpecification:
    """Return an ensemble with the same members and an explicitly locked weight."""
    if specification.model_kind is not ModelKind.ENSEMBLE or specification.ensemble is None:
        raise ValueError("invalid_model_specification")
    return ensemble_model_specification(
        specification.ensemble.neural_network,
        specification.ensemble.xgboost,
        neural_network_weight,
    )


def specification_parameters(specification: ModelSpecification) -> dict[str, Any]:
    """Return framework parameters while keeping their typed specification source."""
    if specification.model_kind is ModelKind.ENSEMBLE:
        raise ValueError("ensemble_has_no_component_parameters")
    return {
        **_jsonable(specification.architecture_parameters),
        **_jsonable(specification.training_parameters),
    }


def fixed_model_specification() -> ModelSpecification:
    """Return the historical fixed NN/XGBoost/ensemble as ordinary specifications."""
    neural = ModelSpecification(
        ModelKind.NEURAL_NETWORK,
        PreprocessingContract("float32", "fit_on_training_records"),
        {
            "layers": (448, 601, 544, 416),
            "layer_specs": ((448, "selu", 0.0), (601, "mish", 0.3),
                            (544, "mish", 0.3), (416, "selu", 0.3)),
            "activation": "mish", "dropout": 0.3, "optimizer": "adamw",
            "loss": "mean_squared_error", "l2": 3.436477039390904e-06,
        },
        {
            "learning_rate": 0.0006350310563507329, "batch_size": 256,
            "maximum_epochs": 100, "early_stopping_patience": 8,
            "early_stopping_min_delta": 0.005, "lr_reduction_factor": 0.5,
            "lr_reduction_patience": 3, "minimum_learning_rate": 1e-8,
        },
    )
    xgboost = ModelSpecification(
        ModelKind.XGBOOST,
        PreprocessingContract("float32", "none"),
        {
            "n_estimators": 900, "max_depth": 9, "learning_rate": 0.01,
            "subsample": 0.7, "colsample_bytree": 0.9, "min_child_weight": 1.0,
            "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
            "objective": "reg:squarederror", "n_jobs": 1,
        },
        {"early_stopping_rounds": 50},
    )
    return ensemble_model_specification(neural, xgboost, 0.2)


class TensorflowXGBoostBackend:
    """Framework adapter; all TensorFlow/XGBoost construction lives in this module."""

    def __init__(self) -> None:
        if not ((3, 11) <= sys.version_info[:2] <= (3, 13)):
            raise RuntimeError("unsupported_model_runtime")
        import keras  # type: ignore[import-not-found]
        import numpy as np
        import tensorflow as tf  # type: ignore[import-untyped]
        import xgboost  # type: ignore[import-not-found]
        from .learned import enable_synchronous_dataset_execution
        if not enable_synchronous_dataset_execution():
            raise RuntimeError("synchronous_tensorflow_dataset_runtime_required")
        self.np = np
        self.tf = tf
        self.xgboost = xgboost
        self.dependency_versions: dict[str, str] = {
            "keras": keras.__version__, "numpy": np.__version__,
            "tensorflow": tf.__version__, "xgboost": xgboost.__version__,
        }

    def fit_component(
        self, specification: ModelSpecification, training: TrainingData,
        validation: ValidationData | None, seed: int,
    ) -> FittedModel:
        if specification.model_kind is ModelKind.NEURAL_NETWORK:
            return self._fit_neural(specification, training, validation, seed)
        if specification.model_kind is ModelKind.XGBOOST:
            return self._fit_xgboost(specification, training, validation, seed)
        raise ValueError("unsupported_model_kind")

    def serialize_component(self, fitted: FittedModel) -> Mapping[str, bytes]:
        model = fitted.backend_state.get("model")
        if model is None:
            raise ValueError("model_artifact_unavailable")
        with tempfile.TemporaryDirectory() as directory:
            if fitted.specification.model_kind is ModelKind.XGBOOST:
                path = Path(directory) / "model.json"
                model.save_model(path)
            else:
                path = Path(directory) / "model.keras"
                model.save(path, include_optimizer=False)
            return {path.name: path.read_bytes()}

    def load_component(
        self, specification: ModelSpecification, artifacts: Mapping[str, bytes],
        preprocessing_state: Mapping[str, Any],
    ) -> BatchPredictor:
        del preprocessing_state
        np, tf = self.np, self.tf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if specification.model_kind is ModelKind.NEURAL_NETWORK:
                content = artifacts.get("model.keras")
                if content is None:
                    raise ValueError("model_artifact_unavailable")
                path = root / "model.keras"
                path.write_bytes(content)
                model = tf.keras.models.load_model(path, compile=False)
                return lambda rows: np.asarray(
                    model(np.asarray(rows, dtype=np.float32), training=False)
                ).reshape(-1).tolist()
            if specification.model_kind is ModelKind.XGBOOST:
                content = artifacts.get("model.json")
                if content is None:
                    raise ValueError("model_artifact_unavailable")
                path = root / "model.json"
                path.write_bytes(content)
                model = self.xgboost.XGBRegressor()
                model.load_model(path)
                return lambda rows: model.predict(
                    np.asarray(rows, dtype=np.float32)
                ).reshape(-1).tolist()
        raise ValueError("unsupported_model_kind")

    def _fit_neural(
        self, specification: ModelSpecification, training: TrainingData,
        validation: ValidationData | None, seed: int,
    ) -> FittedModel:
        np, tf = self.np, self.tf
        parameters = specification_parameters(specification)
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except RuntimeError:
            pass
        train_x = np.asarray(training.features, dtype=np.float32)
        train_y = np.asarray(training.targets, dtype=np.float32)
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(train_x)
        model = tf.keras.Sequential([tf.keras.Input(shape=(len(specification.ordered_prediction_features),)), normalizer])
        regularizer = tf.keras.regularizers.l2(float(parameters["l2"]))
        layer_specs = parameters.get("layer_specs") or tuple(
            (units, parameters["activation"], parameters["dropout"])
            for units in parameters["layers"]
        )
        for units, activation, dropout in layer_specs:
            model.add(tf.keras.layers.Dense(int(units), kernel_regularizer=regularizer))
            model.add(tf.keras.layers.Activation(activation))
            if dropout:
                model.add(tf.keras.layers.Dropout(float(dropout)))
        model.add(tf.keras.layers.Dense(1))
        optimizer_class = (
            tf.keras.optimizers.AdamW if parameters["optimizer"] == "adamw"
            else tf.keras.optimizers.Adam
        )
        model.compile(
            optimizer=optimizer_class(float(parameters["learning_rate"])),
            loss=parameters["loss"], metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )
        kwargs: dict[str, Any] = {
            "verbose": 0, "shuffle": True,
            "epochs": int(parameters["maximum_epochs"]),
            "batch_size": int(parameters["batch_size"]),
        }
        if validation is not None:
            validation_x = np.asarray(validation.features, dtype=np.float32)
            validation_y = np.asarray(validation.targets, dtype=np.float32)
            kwargs["validation_data"] = (validation_x, validation_y)
            callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor="val_mean_absolute_error", mode="min",
                min_delta=float(parameters.get("early_stopping_min_delta", 0.0)),
                patience=int(parameters["early_stopping_patience"]),
                restore_best_weights=True, verbose=0,
            )]
            if "lr_reduction_factor" in parameters:
                callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_mean_absolute_error", mode="min",
                    factor=float(parameters["lr_reduction_factor"]),
                    patience=int(parameters["lr_reduction_patience"]),
                    min_lr=float(parameters["minimum_learning_rate"]), verbose=0,
                ))
            kwargs["callbacks"] = callbacks
        history = model.fit(train_x, train_y, **kwargs)
        selected_epochs = len(history.epoch)
        if validation is not None:
            selected_epochs = _selected_epoch_count(
                history.history.get("val_mean_absolute_error", ()),
                float(parameters.get("early_stopping_min_delta", 0.0)),
            )
        parameter_hash = sha256()
        for weights in model.get_weights():
            parameter_hash.update(np.asarray(weights).tobytes())
        normalization = {
            "mean": normalizer.mean.numpy().reshape(-1).tolist(),
            "variance": normalizer.variance.numpy().reshape(-1).tolist(),
        }
        return FittedModel(
            specification,
            lambda rows: np.asarray(
                model(np.asarray(rows, dtype=np.float32), training=False)
            ).reshape(-1).tolist(),
            normalization,
            {"selected_epochs": selected_epochs,
             "fitted_parameter_fingerprint": parameter_hash.hexdigest()},
            {"model": model},
        )

    def _fit_xgboost(
        self, specification: ModelSpecification, training: TrainingData,
        validation: ValidationData | None, seed: int,
    ) -> FittedModel:
        np = self.np
        parameters = specification_parameters(specification)
        early_stopping = parameters.pop("early_stopping_rounds")
        model = self.xgboost.XGBRegressor(
            **parameters, random_state=seed, tree_method="hist", eval_metric="mae",
            **({"early_stopping_rounds": early_stopping} if validation is not None else {}),
        )
        fit_kwargs = {}
        if validation is not None:
            fit_kwargs = {
                "eval_set": [(np.asarray(validation.features, dtype=np.float32),
                              np.asarray(validation.targets, dtype=np.float32))],
                "verbose": False,
            }
        model.fit(
            np.asarray(training.features, dtype=np.float32),
            np.asarray(training.targets, dtype=np.float32), **fit_kwargs,
        )
        selected = (
            int(model.best_iteration) + 1 if validation is not None
            else int(parameters["n_estimators"])
        )
        return FittedModel(
            specification,
            lambda rows: model.predict(np.asarray(rows, dtype=np.float32)).reshape(-1).tolist(),
            {"xgboost": "unnormalized_float32"},
            {"selected_trees": selected, "fitted_parameter_fingerprint": sha256(
                bytes(model.get_booster().save_raw())
            ).hexdigest()},
            {"model": model},
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


def _validate_neural_candidate(parameters: Mapping[str, Any]) -> None:
    expected = {
        "layers", "activation", "dropout", "optimizer", "loss", "learning_rate",
        "l2", "batch_size", "maximum_epochs", "early_stopping_patience",
    }
    if set(parameters) != expected:
        raise ValueError("invalid_model_specification")
    _validate_neural_parameters(parameters)


def _validate_neural_parameters(parameters: Mapping[str, Any]) -> None:
    required = {
        "layers", "activation", "dropout", "optimizer", "loss", "learning_rate",
        "l2", "batch_size", "maximum_epochs", "early_stopping_patience",
    }
    optional = {
        "layer_specs", "early_stopping_min_delta", "lr_reduction_factor",
        "lr_reduction_patience", "minimum_learning_rate",
    }
    if not required.issubset(parameters) or not set(parameters).issubset(required | optional):
        raise ValueError("invalid_model_specification")
    layers = parameters["layers"]
    if (
        not isinstance(layers, (tuple, list)) or not layers
        or any(not _positive_int(value) for value in layers)
        or parameters["activation"] not in {"relu", "selu", "mish"}
        or parameters["optimizer"] not in {"adam", "adamw"}
        or parameters["loss"] not in {"mean_absolute_error", "huber", "mean_squared_error"}
        or not _bounded_number(parameters["dropout"], 0.0, 1.0, upper_inclusive=False)
        or not _positive_number(parameters["learning_rate"])
        or not _bounded_number(parameters["l2"], 0.0, math.inf)
        or not _positive_int(parameters["batch_size"])
        or not _positive_int(parameters["maximum_epochs"])
        or not _positive_int(parameters["early_stopping_patience"])
    ):
        raise ValueError("invalid_model_specification")
    layer_specs = parameters.get("layer_specs")
    if layer_specs is not None and (
        not isinstance(layer_specs, (tuple, list)) or len(layer_specs) != len(layers)
        or any(
            not isinstance(item, (tuple, list)) or len(item) != 3
            or not _positive_int(item[0])
            or item[1] not in {"relu", "selu", "mish"}
            or not _bounded_number(item[2], 0.0, 1.0, upper_inclusive=False)
            for item in layer_specs
        )
    ):
        raise ValueError("invalid_model_specification")
    optional_numbers = {
        "early_stopping_min_delta": lambda value: _bounded_number(value, 0.0, math.inf),
        "lr_reduction_factor": lambda value: _bounded_number(
            value, 0.0, 1.0, upper_inclusive=False
        ),
        "lr_reduction_patience": _positive_int,
        "minimum_learning_rate": _positive_number,
    }
    if any(name in parameters and not check(parameters[name])
           for name, check in optional_numbers.items()):
        raise ValueError("invalid_model_specification")


def _validate_xgboost(parameters: Mapping[str, Any]) -> None:
    expected = {
        "n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
        "min_child_weight", "gamma", "reg_alpha", "reg_lambda", "objective", "n_jobs",
        "early_stopping_rounds",
    }
    if (
        set(parameters) != expected
        or not _positive_int(parameters["n_estimators"])
        or not _positive_int(parameters["max_depth"])
        or not _positive_number(parameters["learning_rate"])
        or not _bounded_number(parameters["subsample"], 0.0, 1.0)
        or not _bounded_number(parameters["colsample_bytree"], 0.0, 1.0)
        or not _positive_number(parameters["min_child_weight"])
        or not _bounded_number(parameters["gamma"], 0.0, math.inf)
        or not _bounded_number(parameters["reg_alpha"], 0.0, math.inf)
        or not _positive_number(parameters["reg_lambda"])
        or parameters["objective"] != "reg:squarederror"
        or parameters["n_jobs"] != 1
        or not _positive_int(parameters["early_stopping_rounds"])
    ):
        raise ValueError("invalid_model_specification")


def _ensemble_predictor(
    specification: ModelSpecification,
    neural: BatchPredictor,
    xgboost: BatchPredictor,
) -> BatchPredictor:
    def predict(rows: Sequence[tuple[float, ...]]) -> list[float]:
        left = tuple(float(value) for value in neural(rows))
        right = tuple(float(value) for value in xgboost(rows))
        if len(left) != len(rows) or len(right) != len(rows):
            raise ValueError("invalid_model_prediction")
        return list(combine_ensemble_predictions(specification, left, right))
    return predict


def _immutable_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(_plain_mapping(value))


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid_model_specification")
    return {str(key): _freeze(item) for key, item in value.items()}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _validate_data(
    features: Sequence[tuple[float, ...]], targets: Sequence[float]
) -> None:
    if len(features) != len(targets) or not features:
        raise ValueError("invalid_model_training_data")
    if any(not row or any(not _finite_number(value) for value in row) for row in features):
        raise ValueError("invalid_model_training_data")
    if any(not _finite_number(value) for value in targets):
        raise ValueError("invalid_model_training_data")


def _validate_feature_width(
    features: Sequence[tuple[float, ...]], expected: int,
) -> None:
    if any(len(row) != expected for row in features):
        raise ValueError("invalid_model_feature_width")


def _validated_artifacts(artifacts: Mapping[str, bytes]) -> dict[str, bytes]:
    if not artifacts or any(
        not isinstance(name, str) or not name or "/" in name or "\\" in name
        or not isinstance(content, bytes)
        for name, content in artifacts.items()
    ):
        raise ValueError("invalid_model_artifacts")
    return dict(artifacts)


def _strip_prefix(artifacts: Mapping[str, bytes], prefix: str) -> dict[str, bytes]:
    selected = {name[len(prefix):]: value for name, value in artifacts.items()
                if name.startswith(prefix)}
    return _validated_artifacts(selected)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_number(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _bounded_number(
    value: Any, lower: float, upper: float, *, upper_inclusive: bool = True
) -> bool:
    return (
        _finite_number(value) and float(value) >= lower
        and (float(value) <= upper if upper_inclusive else float(value) < upper)
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
