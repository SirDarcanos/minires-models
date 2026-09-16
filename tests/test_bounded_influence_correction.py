"""Synthetic numeric tests for the NEW bounded-influence loss (not old tail loss)."""
import copy
import math
import unittest

import numpy as np

from minires.modeling import bounded_influence_correction as numeric
from minires.modeling import tail_correction as old
from minires.modeling.definitions import (
    combine_ensemble_predictions, ensemble_with_weight, fixed_model_specification,
)


class BoundedInfluenceCorrectionTests(unittest.TestCase):
    def test_real_prediction_derivative_matches_finite_differences_including_boundaries(self):
        def loss(value):
            ordinary, excess = numeric.prediction_loss_components([value])
            return float(ordinary[0] + excess[0])
        for error in (-100, -7, -5, -4.5, -4, 0, 4, 4.5, 5, 7, 100):
            with self.subTest(error=error):
                h = 1e-6
                difference = (loss(error + h) - loss(error - h)) / (2 * h)
                self.assertAlmostEqual(float(numeric.prediction_derivative([error])[0]), difference, places=5)
        ordinary, excess = numeric.prediction_loss_components([4.5, 5, 7, 10])
        np.testing.assert_allclose(ordinary, [2.025, 2.5, 4.5, 7.5])
        np.testing.assert_allclose(excess, [0, 1, 25, 85])

    def test_full_convex_objective_gradient_matches_finite_difference(self):
        design = np.array([[1, -1, 0.5, 1], [1, 0.1, -0.3, 0], [1, 0, 1, -1]])
        coefs = np.array([0.1, -0.1, 0.2, 0.3])
        initial_errors = np.array([-8, 4.5, 5])
        def objective(c):
            correction = design @ c
            ordinary, excess = numeric.prediction_loss_components(initial_errors + correction)
            return np.mean(ordinary + excess + correction ** 2) + 0.01 * np.sum(c ** 2)
        correction = design @ coefs
        gradient = design.T @ (numeric.prediction_derivative(initial_errors + correction) + 2 * correction) / 3 + 0.02 * coefs
        for i in range(4):
            delta = np.eye(4)[i] * 1e-6
            self.assertAlmostEqual(gradient[i], (objective(coefs + delta) - objective(coefs - delta)) / 2e-6, places=7)

    def test_both_components_have_bounded_influence_unlike_old_squared_objective(self):
        errors = np.array([-3e38, -1e20, -100, -7, 7, 100, 1e20, 3e38])
        derivative = numeric.prediction_derivative(errors)
        np.testing.assert_array_equal(derivative, [-21] * 4 + [21] * 4)
        old_derivative = 0.2 * errors + 8 * np.sign(errors) * np.maximum(np.abs(errors) - 4.5, 0)
        self.assertGreater(abs(old_derivative[-1]), 1e38)
        # Contrast the real old/new fitted objectives with identical synthetic rows.
        bases = ((10.0,) * 100, (10.0,) * 100)
        new_moderate = numeric.fit_state(bases, (10.0,) * 99 + (30.0,))
        new_extreme = numeric.fit_state(bases, (10.0,) * 99 + (1000000.0,))
        np.testing.assert_array_equal(new_moderate["coefficients"], new_extreme["coefficients"])
        old_moderate = old.fit_state(bases, (10.0,) * 99 + (30.0,))
        old_extreme = old.fit_state(bases, (10.0,) * 99 + (1000000.0,))
        self.assertGreater(old_extreme["coefficients"][0], old_moderate["coefficients"][0])

    def test_exact_identity_determinism_and_global_projected_bound(self):
        bases = ((1.25, 9.5, 100.125), (31.75, 8.125, 2.75))
        targets = (1e5, -1e5, 1e5)
        state = numeric.fit_state(bases, targets)
        self.assertEqual(state, numeric.fit_state(bases, targets))
        self.assertLessEqual(math.fsum(abs(v) for v in state["coefficients"]), 2)
        expected = combine_ensemble_predictions(
            ensemble_with_weight(fixed_model_specification(), 0.8), *bases)
        self.assertEqual(numeric.predict(state, bases, 0)[0], expected)
        self.assertEqual(numeric.predict(state, bases, 0)[1], (0.,) * 3)
        for column in (bases, ((1e20, -1e20), (-1e20, 1e20))):
            _, corrections = numeric.predict(state, column, 1)
            self.assertTrue(all(abs(value) <= 2 for value in corrections))
        # Every sign corner including intercept is bounded by the L1 projection.
        for sign in np.ndindex(2, 2, 2, 2):
            projected = numeric._project(np.array([100 if v else -100 for v in sign], dtype=float))
            self.assertLessEqual(math.fsum(abs(v) for v in projected), 2)

    def test_typed_state_tampering_and_old_new_loader_separation(self):
        bases = ((10.,), (20.,))
        state = numeric.fit_state(bases, (12.,))
        self.assertFalse(old.valid_state(state))
        self.assertFalse(numeric.valid_state(old.fit_state(bases, (12.,))))
        mutations = [
            lambda s: s.update(coefficients=[3., 0., 0., 0.]),
            lambda s: s.update(coefficients=[True, 0., 0., 0.]),
            lambda s: s.update(feature_means=[float("nan"), 0., 0.]),
            lambda s: s.update(feature_means=[4e38, 0., 0.]),
            lambda s: s.update(feature_scales=[0., 1., 1.]),
            lambda s: s.update(feature_scales=[1., 1.]),
            lambda s: s.update(feature_means=[[0.], [0.], [0.]]),
            lambda s: s.update(extra="unexpected"),
            lambda s: s["contract"].update(iterations=2001),
            lambda s: s["contract"].update(ordinary_huber_delta_g=6.),
            lambda s: s["contract"].update(excess_huber_delta_g=3.),
            lambda s: s["contract"].update(intercept=1),
            lambda s: s["contract"].update(features=tuple(numeric.FEATURES)),
            lambda s: s["contract"].update(iterations=2000.0),
            lambda s: s.update(coefficients=[10 ** 1000, 0., 0., 0.]),
        ]
        for mutate in mutations:
            altered = copy.deepcopy(state)
            mutate(altered)
            self.assertFalse(numeric.valid_state(altered))
            with self.assertRaises(ValueError):
                numeric.predict(altered, bases, 0)
        for invalid in (None, [], {}, {"contract": object()}):
            self.assertFalse(numeric.valid_state(invalid))
        self.assertEqual(numeric.fit_state(bases, (12.,)), state)

    def test_invalid_shapes_nonfinite_float32_limits_and_overflow_fail_closed(self):
        for bases, targets in (([[], []], []), ([[0], [0]], [[0]]), ([[0], [0]], [0, 0]),
                               ([[4e38], [0]], [0]), ([[0], [0]], [4e38]),
                               ([[3e38], [-3e38]], [0]), ([[math.nan], [0]], [0]),
                               ([[0], [0]], [math.inf])):
            with self.subTest(bases=bases, targets=targets), self.assertRaises((ValueError, OverflowError)):
                numeric.fit_state(bases, targets)
        for error in ([], [[1]], [math.inf], [math.nan], [4e38], [1e309]):
            for function in (numeric.prediction_derivative, numeric.prediction_loss_components):
                with self.assertRaises(ValueError):
                    function(error)
        state = numeric.fit_state(((0.,), (0.,)), (0.,))
        state["feature_scales"] = [1e-320] * 3
        with self.assertRaises(ValueError):
            numeric.predict(state, ((1e20,), (0.,)), 1)
        with self.assertRaises(ValueError):
            numeric.predict(state, ((0.,), (0.,)), True)


if __name__ == "__main__":
    unittest.main()
