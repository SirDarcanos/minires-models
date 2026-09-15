"""Public fit/predict contract for the frozen closest-anchor correction."""
import copy
import unittest

from minires.modeling.tail_correction import fit_state, predict
from minires.modeling.definitions import (
    combine_ensemble_predictions, ensemble_with_weight, fixed_model_specification,
)


class TailCorrectionTests(unittest.TestCase):
    def test_frozen_anchor_identity_and_bounded_tail_learning(self):
        bases = ((10.0,) * 20, (20.0,) * 20)
        state = fit_state(bases, (20.0,) * 20)
        identity, zero = predict(state, bases, 0.0)
        self.assertEqual(identity, (12.0,) * 20)
        self.assertEqual(zero, (0.0,) * 20)
        corrected, correction = predict(state, bases, 1.0)
        self.assertTrue(all(13.9 < value <= 14.0 for value in corrected))
        self.assertTrue(all(0.0 < value <= 2.0 for value in correction))
        self.assertEqual(state, fit_state(bases, (20.0,) * 20))

    def test_identity_preserves_original_ensemble_float64_operation_order(self):
        bases = ((1.25, 9.5, 100.125), (31.75, 8.125, 2.75))
        state = fit_state(bases, (7.35, 9.225, 80.65))
        expected = combine_ensemble_predictions(
            ensemble_with_weight(fixed_model_specification(), 0.8), bases[0], bases[1]
        )
        self.assertEqual(predict(state, bases, 0.0)[0], expected)
        state["contract"]["features"][0] = "modified"
        with self.assertRaises(ValueError):
            predict(state, bases, 0.0)
        self.assertEqual(fit_state(bases, (7.35, 9.225, 80.65))["contract"]["features"][0], "anchor_g")

    def test_unclipped_severe_errors_retain_more_influence_than_moderate_errors(self):
        bases = ((10.0,) * 100, (10.0,) * 100)
        moderate = fit_state(bases, (10.0,) * 99 + (16.0,))
        severe = fit_state(bases, (10.0,) * 99 + (30.0,))
        self.assertGreater(predict(severe, bases, 1.0)[1][0],
                           3 * predict(moderate, bases, 1.0)[1][0])

    def test_nonfinite_and_modified_states_fail_closed_even_for_identity(self):
        bases = ((10.0,), (20.0,))
        state = fit_state(bases, (12.0,))
        for key, value in (("coefficients", [3.0, 0.0, 0.0, 0.0]),
                           ("feature_scales", [0.0, 1.0, 1.0]),
                           ("feature_means", [float("nan"), 0.0, 0.0])):
            altered = copy.deepcopy(state)
            altered[key] = value
            with self.assertRaises(ValueError):
                predict(altered, bases, 0.0)
        for invalid in (((float("nan"),), (20.0,)), ((10.0,), ()),
                        ((4e38,), (20.0,))):
            with self.assertRaises(ValueError):
                predict(state, invalid, 1.0)
        altered = copy.deepcopy(state)
        altered["contract"]["iterations"] = 2001
        with self.assertRaises(ValueError):
            predict(altered, bases, 1.0)


if __name__ == "__main__":
    unittest.main()
