import importlib.util
import math
import unittest

from minires.modeling import bounded_tail_risk


class BoundedTailRiskLossTests(unittest.TestCase):
    def test_loss_has_predeclared_known_values(self):
        self.assertEqual(bounded_tail_risk.loss_value(0.0), 0.0)
        self.assertAlmostEqual(bounded_tail_risk.loss_value(2.0), 0.4)
        self.assertAlmostEqual(bounded_tail_risk.loss_value(5.0), 3.5)
        self.assertAlmostEqual(bounded_tail_risk.loss_value(7.0), 29.5)
        self.assertAlmostEqual(bounded_tail_risk.loss_value(-7.0), 29.5)

    def test_loss_gradient_is_finite_odd_and_bounded(self):
        for error in (-1e300, -7.0, -5.0, -4.5, 0.0, 4.5, 5.0, 7.0, 1e300):
            gradient = bounded_tail_risk.loss_gradient(error)
            self.assertTrue(math.isfinite(gradient))
            self.assertLessEqual(abs(gradient), bounded_tail_risk.MAXIMUM_ABSOLUTE_GRADIENT)
            self.assertAlmostEqual(gradient, -bounded_tail_risk.loss_gradient(-error))

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "tensorflow optional")
    def test_tensorflow_loss_matches_scalar_contract_and_has_bounded_gradient(self):
        import tensorflow as tf

        loss = bounded_tail_risk.tensorflow_loss(tf)
        for error in (-1e30, -20.0, -7.0, -5.0, 0.0, 5.0, 7.0, 20.0, 1e30):
            prediction = tf.Variable([[error]], dtype=tf.float32)
            with tf.GradientTape() as tape:
                observed = loss(tf.constant([[0.0]]), prediction)
            gradient = float(tape.gradient(observed, prediction).numpy()[0, 0])
            expected = bounded_tail_risk.loss_value(error)
            self.assertTrue(math.isfinite(float(observed.numpy())))
            self.assertTrue(math.isclose(float(observed.numpy()), expected, rel_tol=1e-6, abs_tol=1e-5))
            self.assertAlmostEqual(gradient, bounded_tail_risk.loss_gradient(error), places=5)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "tensorflow optional")
    def test_tensorflow_loss_is_finite_at_float32_extremes(self):
        import numpy as np
        import tensorflow as tf

        maximum = np.finfo(np.float32).max
        loss = bounded_tail_risk.tensorflow_loss(tf)
        prediction = tf.Variable([[-maximum]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            observed = loss(tf.constant([[maximum]], dtype=tf.float32), prediction)
        gradient = tape.gradient(observed, prediction)

        self.assertEqual(observed.dtype, tf.float64)
        self.assertTrue(math.isfinite(float(observed.numpy())))
        self.assertIsNotNone(gradient)
        self.assertTrue(math.isfinite(float(gradient.numpy()[0, 0])))
        self.assertLessEqual(
            abs(float(gradient.numpy()[0, 0])),
            bounded_tail_risk.MAXIMUM_ABSOLUTE_GRADIENT,
        )

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "tensorflow optional")
    def test_custom_loss_model_round_trips_with_compile_free_loading(self):
        from minires.modeling.definitions import (
            ModelRuntime, TensorflowXGBoostBackend, TrainingData,
            candidate_model_specification,
        )

        parameters = dict(bounded_tail_risk.base_candidates()[1].parameters)
        parameters["maximum_epochs"] = 1
        specification = candidate_model_specification(
            "neural_network", parameters,
            identity_namespace=bounded_tail_risk.VERSION,
        )
        runtime = ModelRuntime(TensorflowXGBoostBackend())
        rows = tuple((float(index),) * 7 for index in range(1, 5))
        fitted = runtime.fit(
            specification, TrainingData(rows, (1.0, 2.0, 3.0, 4.0)), None, seed=41,
        )
        artifacts = runtime.save(fitted)
        loaded = runtime.load_verified_component(
            specification, artifacts, fitted.preprocessing_state,
        )
        predictions = loaded(rows)
        self.assertEqual(len(predictions), len(rows))
        self.assertTrue(all(math.isfinite(float(value)) for value in predictions))

    def test_fixed_candidates_change_only_the_neural_loss(self):
        legacy, tail, xgboost = bounded_tail_risk.base_candidates()
        expected_neural = {
            "activation": "mish", "batch_size": 256, "dropout": 0.1,
            "early_stopping_patience": 8, "l2": 1e-5,
            "layers": (256, 128, 64), "learning_rate": 0.003,
            "maximum_epochs": 100, "optimizer": "adam",
            "validation_selection": "serious_error_gates_then_ranking_v1",
        }
        self.assertEqual({key: legacy.parameters[key] for key in expected_neural}, expected_neural)
        self.assertEqual({key: tail.parameters[key] for key in expected_neural}, expected_neural)
        self.assertEqual(legacy.parameters["loss"], "huber")
        self.assertEqual(tail.parameters["loss"], bounded_tail_risk.LOSS_NAME)
        self.assertEqual(xgboost.parameters["n_estimators"], 1200)
        self.assertEqual(
            bounded_tail_risk.FIXED_COUNTS,
            {"neural_network_epochs": 87, "xgboost_trees": 1091},
        )
        self.assertEqual(xgboost.parameters["max_depth"], 9)
        self.assertEqual(xgboost.parameters["learning_rate"], 0.05)
        self.assertEqual(xgboost.parameters["subsample"], 0.9)
        self.assertEqual(xgboost.parameters["colsample_bytree"], 0.9)
        self.assertEqual(xgboost.parameters["min_child_weight"], 10.0)
        self.assertEqual(xgboost.parameters["gamma"], 0.2)
        self.assertEqual(xgboost.parameters["reg_alpha"], 0.1)
        self.assertEqual(xgboost.parameters["reg_lambda"], 10.0)

    def test_loss_rejects_nonfinite_errors(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                bounded_tail_risk.loss_value(value)
            with self.assertRaises(ValueError):
                bounded_tail_risk.loss_gradient(value)


if __name__ == "__main__":
    unittest.main()
