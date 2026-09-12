import unittest

from minires.modeling.definitions import (
    FittedModel,
    ModelKind,
    ModelRuntime,
    TrainingData,
    ValidationData,
    candidate_model_specification,
    ensemble_model_specification,
    fixed_model_specification,
)
from minires.modeling.tuning import DeclaredCandidate


NEURAL_PARAMETERS = {
    "layers": (64, 32),
    "activation": "relu",
    "dropout": 0.1,
    "optimizer": "adam",
    "loss": "mean_absolute_error",
    "learning_rate": 0.001,
    "l2": 0.00001,
    "batch_size": 32,
    "maximum_epochs": 100,
    "early_stopping_patience": 8,
}

XGBOOST_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.75,
    "colsample_bytree": 0.9,
    "min_child_weight": 3.0,
    "gamma": 0.05,
    "reg_alpha": 0.001,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": 1,
    "early_stopping_rounds": 50,
}


class RecordingBackend:
    dependency_versions = {"backend": "test-1"}

    def __init__(self):
        self.fit_calls = []
        self.load_calls = []

    def fit_component(self, specification, training, validation, seed):
        self.fit_calls.append((specification, training, validation, seed))
        offset = 1.0 if specification.model_kind is ModelKind.NEURAL_NETWORK else 3.0
        return FittedModel(
            specification=specification,
            predictor=lambda rows: [float(row[0]) + offset for row in rows],
            preprocessing_state={"kind": specification.model_kind.value},
            metadata={"selected_epochs": 4} if offset == 1.0 else {"selected_trees": 20},
            backend_state={"kind": specification.model_kind.value},
        )

    def serialize_component(self, fitted):
        return {"model.bin": fitted.specification.model_kind.value.encode()}

    def load_component(self, specification, artifacts, preprocessing_state):
        self.load_calls.append((specification, artifacts, preprocessing_state))
        offset = 1.0 if specification.model_kind is ModelKind.NEURAL_NETWORK else 3.0
        return lambda rows: [float(row[0]) + offset for row in rows]


class ModelSpecificationTests(unittest.TestCase):
    def test_candidate_specification_is_validated_inspectable_and_stable(self):
        first = candidate_model_specification("neural_network", NEURAL_PARAMETERS)
        second = candidate_model_specification(
            "neural_network", dict(reversed(tuple(NEURAL_PARAMETERS.items())))
        )

        self.assertEqual(first.stable_identity, second.stable_identity)
        self.assertEqual(first.model_kind, ModelKind.NEURAL_NETWORK)
        self.assertEqual(first.ordered_prediction_features[0], "kb")
        self.assertEqual(first.preprocessing.normalization, "fit_on_training_records")
        self.assertEqual(first.architecture_parameters["layers"], (64, 32))
        self.assertEqual(first.training_parameters["maximum_epochs"], 100)
        self.assertEqual(first.output_unit, "g")
        self.assertIn("neural_network", first.describe())
        self.assertIn("64 → 32 → 1", first.describe())
        self.assertEqual(
            DeclaredCandidate("neural_network", NEURAL_PARAMETERS).candidate_id,
            "declared-neural_network-3bf2533b7032467a",
        )

        with self.assertRaisesRegex(ValueError, "invalid_model_specification"):
            candidate_model_specification(
                "xgboost", {**XGBOOST_PARAMETERS, "n_jobs": -1}
            )

    def test_fixed_configuration_uses_the_same_interface_as_candidates(self):
        fixed = fixed_model_specification()

        self.assertEqual(fixed.model_kind, ModelKind.ENSEMBLE)
        self.assertEqual(fixed.ensemble.neural_network_weight, 0.2)
        self.assertEqual(
            fixed.ensemble.neural_network.architecture_parameters["layers"],
            (448, 601, 544, 416),
        )
        self.assertEqual(
            fixed.ensemble.xgboost.architecture_parameters["n_estimators"], 900
        )
        self.assertTrue(fixed.stable_identity.startswith("model-ensemble-"))


class ModelRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.backend = RecordingBackend()
        self.runtime = ModelRuntime(self.backend)
        self.training = TrainingData(
            ((1.0,) * 7, (2.0,) * 7), (2.0, 3.0)
        )
        self.validation = ValidationData(((4.0,) * 7,), (5.0,))

    def test_fit_and_load_round_trip_components_through_the_public_interface(self):
        specification = candidate_model_specification("xgboost", XGBOOST_PARAMETERS)

        fitted = self.runtime.fit(specification, self.training, self.validation, seed=41)
        artifacts = self.runtime.save(fitted)
        loaded = self.runtime.load(
            specification, artifacts, fitted.preprocessing_state
        )

        self.assertEqual(fitted.predictor(((5.0,) * 7,)), [8.0])
        self.assertEqual(artifacts, {"model.bin": b"xgboost"})
        self.assertEqual(loaded(((5.0,) * 7,)), [8.0])
        self.assertEqual(self.backend.fit_calls[0][1], self.training)
        self.assertEqual(self.backend.fit_calls[0][2], self.validation)

    def test_ensemble_composition_derives_from_its_specification(self):
        neural = candidate_model_specification("neural_network", NEURAL_PARAMETERS)
        xgboost = candidate_model_specification("xgboost", XGBOOST_PARAMETERS)
        specification = ensemble_model_specification(neural, xgboost, 0.25)

        fitted = self.runtime.fit(specification, self.training, self.validation, seed=41)
        artifacts = self.runtime.save(fitted)
        loaded = self.runtime.load(
            specification, artifacts, fitted.preprocessing_state
        )

        self.assertEqual(fitted.predictor(((5.0,) * 7,)), [7.5])
        self.assertEqual(
            artifacts,
            {"neural-model.bin": b"neural_network", "xgboost-model.bin": b"xgboost"},
        )
        self.assertEqual(loaded(((5.0,) * 7,)), [7.5])
        self.assertEqual(len(self.backend.fit_calls), 2)

        legacy_state_loaded = self.runtime.load(
            specification,
            artifacts,
            {
                "neural_network": fitted.preprocessing_state["neural_network"],
                "xgboost": "unnormalized_float32",
            },
        )
        self.assertEqual(legacy_state_loaded(((5.0,) * 7,)), [7.5])

    def test_fit_rejects_data_that_does_not_match_the_feature_contract(self):
        specification = candidate_model_specification("xgboost", XGBOOST_PARAMETERS)

        with self.assertRaisesRegex(ValueError, "invalid_model_feature_width"):
            self.runtime.fit(
                specification,
                TrainingData(((1.0,),), (2.0,)),
                None,
                seed=41,
            )

    def test_public_specification_rejects_missing_kind_parameters(self):
        valid = candidate_model_specification("neural_network", NEURAL_PARAMETERS)

        with self.assertRaisesRegex(ValueError, "invalid_model_specification"):
            type(valid)(
                valid.model_kind,
                valid.preprocessing,
                {},
                {},
            )


if __name__ == "__main__":
    unittest.main()
